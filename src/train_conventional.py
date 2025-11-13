# -*- coding: utf-8 -*-
import os, argparse, json
import numpy as np
import torch, torch.optim as optim
from stgp.utils.io import load_csv_as_tensor
from stgp.models.conventional import nll

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--run_dir", type=str, default="runs/conventional")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--grid_n", type=int, default=50)
    ap.add_argument("--J1", type=int, default=30)
    ap.add_argument("--J2", type=int, default=30)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--q_alpha", type=float, default=1e-4)
    ap.add_argument("--q_beta",  type=float, default=1e-6)
    ap.add_argument("--r_meas",  type=float, default=1.0)
    ap.add_argument("--zeta",    type=float, default=0.1)
    ap.add_argument("--v1", type=float, default=0.0)
    ap.add_argument("--v2", type=float, default=0.0)
    ap.add_argument("--diff1", type=float, default=0.01)
    ap.add_argument("--diff2", type=float, default=0.01)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float64
    torch.set_default_dtype(dtype)

    y = load_csv_as_tensor(args.csv, device, dtype)
    T, m = y.shape
    n = args.grid_n
    assert n*n == m, "grid_n 不匹配 CSV 点数"
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(x, x, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    # learnable params
    v_param = torch.nn.Parameter(torch.tensor([args.v1, args.v2], device=device, dtype=dtype))
    log_diff = torch.nn.Parameter(torch.log(torch.tensor([args.diff1, args.diff2], device=device, dtype=dtype)))
    zeta  = torch.nn.Parameter(torch.tensor(args.zeta, device=device, dtype=dtype))
    log_q_alpha = torch.nn.Parameter(torch.log(torch.tensor(args.q_alpha, device=device, dtype=dtype)))
    log_q_beta  = torch.nn.Parameter(torch.log(torch.tensor(args.q_beta,  device=device, dtype=dtype)))
    log_r_meas  = torch.nn.Parameter(torch.log(torch.tensor(args.r_meas, device=device, dtype=dtype)))

    opt = optim.Adam([v_param, log_diff, zeta, log_q_alpha, log_q_beta, log_r_meas], lr=args.lr)

    for it in range(args.iters):
        opt.zero_grad()
        params = {
            "v_param": v_param, "diff_diag": torch.exp(log_diff),
            "zeta": zeta, "q_alpha": torch.exp(log_q_alpha),
            "q_beta": torch.exp(log_q_beta), "r_meas": torch.exp(log_r_meas)
        }
        loss = nll(y, grid, args.dt, args.J1, args.J2, params, device, dtype, out_dir=args.run_dir if it==0 else None)
        loss.backward()
        opt.step()
        if it % 20 == 0 or it == args.iters-1:
            print(f"[{it:04d}] NLL={float(loss):.3f}")

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "learned_params.json"), "w") as f:
        json.dump({
            'v_param': v_param.detach().cpu().numpy().tolist(),
            'diff_diag': torch.exp(log_diff).detach().cpu().numpy().tolist(),
            'zeta': float(zeta.detach().cpu()),
            'q_alpha': float(torch.exp(log_q_alpha).detach().cpu()),
            'q_beta':  float(torch.exp(log_q_beta).detach().cpu()),
            'r_meas':  float(torch.exp(log_r_meas).detach().cpu())
        }, f, indent=2)
    print("Saved run to:", args.run_dir)

if __name__ == "__main__":
    main()
