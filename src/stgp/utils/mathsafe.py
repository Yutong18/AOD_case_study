import torch

def robust_cholesky(S: torch.Tensor, max_tries: int = 10, device=None, dtype=None):
    I = torch.eye(S.shape[0], device=device or S.device, dtype=dtype or S.dtype)
    S = 0.5 * (S + S.T)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(S if jitter == 0.0 else (S + jitter*I))
            return L
        except RuntimeError:
            jitter = 1e-9 if jitter == 0.0 else jitter * 10.0
    return torch.linalg.cholesky(S + 1e-3*I)
