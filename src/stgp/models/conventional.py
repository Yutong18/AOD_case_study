import os, json
import numpy as np
import torch
from torch.linalg import matrix_exp

def make_K(J1, J2):
    k1 = {(0,0), (0,J2//2), (J1//2,0), (J1//2,J2//2)}
    setA = {(i,j) for i in range(0, J1//2+1) for j in range(0, J2//2+1)}
    setB = {(i,-j) for i in range(1, J1//2) for j in range(1, J2//2)}
    k2 = (setA.union(setB)) - k1
    K_list = sorted(list(k1)) + sorted(list(k2))
    return K_list

def build_F(grid, K, device, dtype):
    m = grid.shape[0]
    n_modes = K.shape[0]
    n_alpha = 2*n_modes
    F = torch.zeros((m, n_alpha), device=device, dtype=dtype)
    for i, k in enumerate(K):
        phase = grid @ k
        F[:, 2*i]     = torch.cos(phase)
        F[:, 2*i + 1] = torch.sin(phase)
    return F

def build_A(dt, K, v_param, diff_diag, zeta, device, dtype):
    D = torch.diag(diff_diag)
    blocks = []
    for k in K:
        lam = -(k @ D @ k) - zeta
        adv =  (v_param @ k)
        g_blk = torch.tensor([[lam, -adv],[adv, lam]], device=device, dtype=dtype)
        G_blk = matrix_exp(g_blk * dt)
        blocks.append(G_blk)
    G_alpha = torch.block_diag(*blocks)
    I = torch.eye(2*K.shape[0], device=device, dtype=dtype)
    top = torch.hstack([G_alpha, I])
    bot = torch.hstack([torch.zeros_like(I), I])
    return torch.vstack([top, bot])

def nll(obs, grid, dt, J1, J2, params, device, dtype, out_dir=None):
    v_param       = params["v_param"]        # shape (2,)
    diff_diag     = params["diff_diag"]      # shape (2,)
    zeta          = params["zeta"]
    q_alpha       = params["q_alpha"]
    q_beta        = params["q_beta"]
    r_meas        = params["r_meas"]

    K_list = make_K(J1, J2)
    K = torch.tensor(K_list, device=device, dtype=dtype)
    K = 2.0*np.pi * K  # frequency (physical units)

    F_alpha = build_F(grid, K, device, dtype)
    F_beta  = F_alpha.clone()

    n_alpha = F_alpha.shape[1]
    A = build_A(dt, K, v_param, diff_diag, zeta, device, dtype)

    Qa = torch.eye(n_alpha, device=device, dtype=dtype) * q_alpha
    Qb = torch.eye(n_alpha, device=device, dtype=dtype) * q_beta
    Q  = torch.block_diag(Qa, Qb)

    R  = torch.eye(F_alpha.shape[0], device=device, dtype=dtype) * r_meas
    H  = torch.hstack([F_alpha, torch.zeros((F_alpha.shape[0], n_alpha), device=device, dtype=dtype)])

    state = torch.zeros(2*n_alpha, device=device, dtype=dtype)
    cov   = torch.eye(2*n_alpha, device=device, dtype=dtype)
    const = F_alpha.shape[0] * torch.log(torch.tensor(2*np.pi, device=device, dtype=dtype))
    total = torch.tensor(0.0, device=device, dtype=dtype)

    for t in range(obs.shape[0]):
        y = obs[t]
        state_pred = A @ state
        cov_pred   = A @ cov @ A.T + Q
        S = H @ cov_pred @ H.T + R
        innov = y - H @ state_pred
        total = total + 0.5*(torch.logdet(S) + innov @ torch.linalg.solve(S, innov) + const)
        K_gain = cov_pred @ H.T @ torch.linalg.solve(S, torch.eye(S.shape[0], device=device, dtype=dtype))
        state = state_pred + K_gain @ innov
        cov   = (torch.eye(2*n_alpha, device=device, dtype=dtype) - K_gain @ H) @ cov_pred

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "K_list.json"), "w") as f:
            json.dump(K_list, f)
    return total
