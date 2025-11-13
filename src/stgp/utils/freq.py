import numpy as np
import torch

def make_omega_sets(Jk_1: int, Jk_2: int):
    Omega1 = {(0, 0), (0, Jk_2//2), (Jk_1//2, 0), (Jk_1//2, Jk_2//2)}
    setA = {(k1, k2) for k1 in range(0, Jk_1//2 + 1)
                    for k2 in range(0, Jk_2//2 + 1)}
    setB = {(k1, k2) for k1 in range(1, Jk_1//2)
                    for k2 in range(-1, -(Jk_2//2), -1)}
    Omega2 = (setA.union(setB)) - Omega1
    return sorted(Omega1), sorted(Omega2)

def build_k_vals(Omega_list, s_max: float, device, dtype):
    if len(Omega_list) == 0:
        return torch.zeros((0, 2), device=device, dtype=dtype)
    arr = np.array(Omega_list, dtype=int)
    k1 = 2.0*np.pi * arr[:, 0] / s_max
    k2 = 2.0*np.pi * arr[:, 1] / s_max
    kv = np.column_stack((k1, k2))
    return torch.tensor(kv, device=device, dtype=dtype)

def construct_F_alpha(s_grid: torch.Tensor,
                      k_vals_omega1: torch.Tensor,
                      k_vals_omega2: torch.Tensor,
                      device, dtype):
    m = s_grid.shape[0]
    n1 = k_vals_omega1.shape[0]
    F_omega1 = torch.zeros((m, 2*n1), device=device, dtype=dtype)
    for i in range(n1):
        phase = s_grid @ k_vals_omega1[i]
        F_omega1[:, 2*i] = torch.cos(phase)  # α_R@Ω1

    n2 = k_vals_omega2.shape[0]
    F_omega2 = torch.zeros((m, 4*n2), device=device, dtype=dtype)
    for i in range(n2):
        phase = s_grid @ k_vals_omega2[i]
        F_omega2[:, 4*i]     = 2.0 * torch.cos(phase)  # α_R@Ω2
        F_omega2[:, 4*i + 2] = 2.0 * torch.sin(phase)  # α_I@Ω2

    return torch.hstack((F_omega1, F_omega2))

def construct_F_beta(s_grid: torch.Tensor,
                     k_vals_omega1: torch.Tensor,
                     k_vals_omega2: torch.Tensor,
                     device, dtype):
    m  = s_grid.shape[0]
    n1 = k_vals_omega1.shape[0]
    n2 = k_vals_omega2.shape[0]

    F1 = torch.zeros((m, 2*n1), device=device, dtype=dtype)
    for i in range(n1):
        phase = s_grid @ k_vals_omega1[i]
        F1[:, 2*i + 1] = torch.cos(phase)              # β_R@Ω1

    F2 = torch.zeros((m, 4*n2), device=device, dtype=dtype)
    for i in range(n2):
        phase = s_grid @ k_vals_omega2[i]
        F2[:, 4*i + 1] = 2.0 * torch.cos(phase)        # β_R@Ω2
        F2[:, 4*i + 3] = 2.0 * torch.sin(phase)        # β_I@Ω2

    return torch.hstack((F1, F2))
