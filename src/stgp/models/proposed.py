import os
import numpy as np
import torch
import torch.nn.functional as F
from stgp.utils.freq import make_omega_sets, build_k_vals, construct_F_alpha, construct_F_beta
from stgp.utils.mathsafe import robust_cholesky

def transform_params(theta: torch.Tensor, device, dtype):
    eps = torch.tensor(1e-6, device=device, dtype=dtype)
    eta       = F.softplus(theta[0]) + eps
    sigma1_sq = F.softplus(theta[1]) + eps
    sigma2_sq = F.softplus(theta[2]) + eps
    mu1, mu2  = theta[3], theta[4]
    l11 = F.softplus(theta[5]) + eps
    l21 = theta[6]
    l22 = F.softplus(theta[7]) + eps
    L   = torch.tensor([[l11, 0.0],[l21, l22]], device=device, dtype=dtype)
    Phi = L @ L.T
    return eta, sigma1_sq, sigma2_sq, mu1, mu2, Phi

def quadform_kPhi(k_vals: torch.Tensor, Phi: torch.Tensor):
    if k_vals.numel() == 0:
        return torch.zeros((0,), device=k_vals.device, dtype=k_vals.dtype)
    return torch.sum((k_vals @ Phi) * k_vals, dim=1)

def build_G_and_Q(dt, k_o1, k_o2, a1, a2, b2, lam1, lam2, device, dtype):
    dt = torch.tensor(dt, device=device, dtype=dtype)
    m1  = dt**3 / 3.0
    m2  = dt + dt**2 + dt**3 / 3.0
    m12 = torch.sqrt(m1*m2)

    blocks = []
    for a in a1:
        blocks.append(torch.tensor([[1.0, dt],[-a*dt, 1.0-dt]], device=device, dtype=dtype))
    G1 = torch.block_diag(*blocks) if blocks else None

    blocks = []
    for a, b in zip(a2, b2):
        blocks.append(torch.tensor([
            [1.0, dt,   0.0, 0.0],
            [-a*dt, 1.0-dt, -b*dt, 0.0],
            [0.0, 0.0, 1.0, dt],
            [b*dt, 0.0, -a*dt, 1.0-dt],
        ], device=device, dtype=dtype))
    G2 = torch.block_diag(*blocks) if blocks else None

    if   G1 is not None and G2 is not None: G = torch.block_diag(G1, G2)
    elif G1 is not None:                    G = G1
    elif G2 is not None:                    G = G2
    else: raise RuntimeError("Empty G")

    blocks = []
    for lam in lam1:
        blocks.append(torch.tensor([[m1*lam, m12*lam],[m12*lam, m2*lam]], device=device, dtype=dtype))
    Q1 = torch.block_diag(*blocks) if blocks else None

    blocks = []
    for lam in lam2:
        blocks.append(torch.tensor([
            [m1*lam, m12*lam, 0.0,    0.0],
            [m12*lam, m2*lam, 0.0,    0.0],
            [0.0,    0.0,     m1*lam, m12*lam],
            [0.0,    0.0,     m12*lam,m2*lam]
        ], device=device, dtype=dtype))
    Q2 = torch.block_diag(*blocks) if blocks else None

    if   Q1 is not None and Q2 is not None: Q = torch.block_diag(Q1, Q2)
    elif Q1 is not None:                    Q = Q1
    elif Q2 is not None:                    Q = Q2
    else: raise RuntimeError("Empty Q")

    return G, Q

def ekf_nll(theta, obs, dt, s_grid, J1, J2, R_param, device, dtype):
    # frequencies
    Om1, Om2 = make_omega_sets(J1, J2)
    k_o1 = build_k_vals(Om1, 1.0, device, dtype)
    k_o2 = build_k_vals(Om2, 1.0, device, dtype)

    # params
    eta, s1, s2, mu1, mu2, Phi = transform_params(theta, device, dtype)
    a1 = eta + 0.5*(s1*(k_o1[:,0]**2) + s2*(k_o1[:,1]**2))
    a2 = eta + 0.5*(s1*(k_o2[:,0]**2) + s2*(k_o2[:,1]**2))
    b2 = mu1*k_o2[:,0] + mu2*k_o2[:,1]
    lam1 = torch.exp(-0.5 * quadform_kPhi(k_o1, Phi))
    lam2 = torch.exp(-0.5 * quadform_kPhi(k_o2, Phi))

    G, Q = build_G_and_Q(dt, k_o1, k_o2, a1, a2, b2, lam1, lam2, device, dtype)

    # meas
    F   = construct_F_alpha(s_grid, k_o1, k_o2, device, dtype)
    R   = torch.eye(F.shape[0], device=device, dtype=dtype) * R_param

    # chunk
    m_points = F.shape[0]
    chunk_sz = 625 if m_points % 625 == 0 else m_points
    chunk_indices = [torch.arange(c*chunk_sz, min((c+1)*chunk_sz, m_points), device=device)
                     for c in range((m_points + chunk_sz - 1) // chunk_sz)]

    # state
    x = torch.zeros(G.shape[0], device=device, dtype=dtype)
    P = torch.eye(G.shape[0], device=device, dtype=dtype) * 100.0

    loglik = torch.tensor(0.0, device=device, dtype=dtype)
    log2pi = torch.log(torch.tensor(2.0*np.pi, device=device, dtype=dtype))

    for t in range(obs.shape[0]):
        x = G @ x
        P = G @ P @ G.T + Q
        y = obs[t]

        for cidx in chunk_indices:
            Fc = F[cidx,:]
            Rc = R[cidx][:,cidx]
            S  = Fc @ P @ Fc.T + Rc
            L  = robust_cholesky(S)
            PFt = P @ Fc.T
            K   = torch.cholesky_solve(PFt.T, L).T
            innov = y[cidx] - Fc @ x
            x = x + K @ innov
            I = torch.eye(P.shape[0], device=device, dtype=dtype)
            P = (I - K @ Fc) @ P @ (I - K @ Fc).T + K @ Rc @ K.T
            P = 0.5*(P + P.T)

            logdet_S = 2.0*torch.sum(torch.log(torch.diag(L)))
            v = torch.cholesky_solve(innov.unsqueeze(1), L).squeeze(1)
            m_c = innov.shape[0]
            loglik = loglik - 0.5*(m_c*log2pi + logdet_S + innov @ v)

    return -loglik  # NLL

@torch.no_grad()
def run_filter(theta, obs, dt, s_grid, J1, J2, R_param, out_dir, device, dtype):
    Om1, Om2 = make_omega_sets(J1, J2)
    k_o1 = build_k_vals(Om1, 1.0, device, dtype)
    k_o2 = build_k_vals(Om2, 1.0, device, dtype)

    eta, s1, s2, mu1, mu2, Phi = transform_params(theta, device, dtype)
    a1 = eta + 0.5*(s1*(k_o1[:,0]**2) + s2*(k_o1[:,1]**2))
    a2 = eta + 0.5*(s1*(k_o2[:,0]**2) + s2*(k_o2[:,1]**2))
    b2 = mu1*k_o2[:,0] + mu2*k_o2[:,1]
    lam1 = torch.exp(-0.5 * quadform_kPhi(k_o1, Phi))
    lam2 = torch.exp(-0.5 * quadform_kPhi(k_o2, Phi))
    G, Q = build_G_and_Q(dt, k_o1, k_o2, a1, a2, b2, lam1, lam2, device, dtype)

    F   = construct_F_alpha(s_grid, k_o1, k_o2, device, dtype)
    Fb  = construct_F_beta (s_grid, k_o1, k_o2, device, dtype)
    R   = torch.eye(F.shape[0], device=device, dtype=dtype) * R_param

    m_points = F.shape[0]
    chunk_sz = 625 if m_points % 625 == 0 else m_points
    chunk_indices = [torch.arange(c*chunk_sz, min((c+1)*chunk_sz, m_points), device=device)
                     for c in range((m_points + chunk_sz - 1) // chunk_sz)]

    x = torch.zeros(G.shape[0], device=device, dtype=dtype)
    P = torch.eye(G.shape[0], device=device, dtype=dtype) * 100.0

    T = obs.shape[0]
    x_rec  = torch.zeros((T, m_points), device=device, dtype=dtype)
    xdot   = torch.zeros((T, m_points), device=device, dtype=dtype)

    n1 = k_o1.shape[0]; n2 = k_o2.shape[0]
    aR1 = torch.zeros((T, n1), device=device, dtype=dtype)
    bR1 = torch.zeros((T, n1), device=device, dtype=dtype)
    aR2 = torch.zeros((T, n2), device=device, dtype=dtype)
    aI2 = torch.zeros((T, n2), device=device, dtype=dtype)
    bR2 = torch.zeros((T, n2), device=device, dtype=dtype)
    bI2 = torch.zeros((T, n2), device=device, dtype=dtype)

    for t in range(T):
        x = G @ x
        P = G @ P @ G.T + Q
        y = obs[t]

        for cidx in chunk_indices:
            Fc = F[cidx,:]
            Rc = R[cidx][:,cidx]
            S  = Fc @ P @ Fc.T + Rc
            L  = robust_cholesky(S)
            PFt = P @ Fc.T
            K   = torch.cholesky_solve(PFt.T, L).T
            innov = y[cidx] - Fc @ x
            x = x + K @ innov
            I = torch.eye(P.shape[0], device=device, dtype=dtype)
            P = (I - K @ Fc) @ P @ (I - K @ Fc).T + K @ Rc @ K.T
            P = 0.5*(P + P.T)

        x_rec[t] = F  @ x
        xdot[t]  = Fb @ x

        if n1 > 0:
            v1 = x[:2*n1]
            aR1[t] = v1[0::2]; bR1[t] = v1[1::2]
        if n2 > 0:
            v2 = x[2*n1:]
            aR2[t] = v2[0::4]; bR2[t] = v2[1::4]
            aI2[t] = v2[2::4]; bI2[t] = v2[3::4]

    aMag1 = aR1.abs(); bMag1 = bR1.abs()
    aMag2 = torch.sqrt(aR2**2 + aI2**2)
    bMag2 = torch.sqrt(bR2**2 + bI2**2)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "x_recon": x_rec.cpu(), "xdot_recon": xdot.cpu(),
        "alpha_R_om1": aR1.cpu(), "beta_R_om1": bR1.cpu(),
        "alpha_R_om2": aR2.cpu(), "alpha_I_om2": aI2.cpu(),
        "beta_R_om2": bR2.cpu(), "beta_I_om2": bI2.cpu(),
        "alpha_mag_om1": aMag1.cpu(), "beta_mag_om1": bMag1.cpu(),
        "alpha_mag_om2": aMag2.cpu(), "beta_mag_om2": bMag2.cpu(),
        "Omega1_list": Om1, "Omega2_list": Om2
    }, os.path.join(out_dir, "proposed_outputs.pt"))
