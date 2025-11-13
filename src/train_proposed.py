import os, argparse, time, json
import numpy as np
import torch, torch.optim as optim
from stgp.utils.io import load_csv_as_tensor, save_json
from stgp.models.proposed import ekf_nll, run_filter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--run_dir", type=str, default="runs/proposed")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--grid_n", type=int, default=50)
    ap.add_argument("--J1", type=int, default=20)
    ap.add_argument("--J2", type=int, default=20)
    ap.add_argument("--R",  type=float, default=0.05**2)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float64
    torch.set_default_dtype(dtype)

    y = load_csv_as_tensor(args.csv, device, dtype)
    T, m = y.shape
    n = args.grid_n
    assert n*n == m, "grid_n not match CSV points"
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(x, x, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    theta = torch.tensor([0.05, 0.01, 0.01, 0.10, 0.00, 0.1, 0.0, 0.1],
                         device=device, dtype=dtype, requires_grad=True)
    opt = optim.Adam([theta], lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20,
                                                     threshold=1e-3, threshold_mode="rel",
                                                     cooldown=5, min_lr=1e-5, verbose=True)

    best = 1e30; best_iter = -1; patience = 80; rel_tol = 1e-4; t0 = time.perf_counter()
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        nll = ekf_nll(theta, y, args.dt, grid, args.J1, args.J2, args.R, device, dtype)
        nll.backward()
        torch.nn.utils.clip_grad_norm_([theta], 100.0)
        opt.step()
        scheduler.step(nll.item())

        if it % 100 == 0 or it == args.iters-1:
            print(f"[{it:04d}] NLL={float(nll):.3f} | lr={opt.param_groups[0]['lr']:.2e} | "
                  f"elapsed={time.perf_counter()-t0:.1f}s")

        if float(nll) < best*(1.0-rel_tol):
            best = float(nll); best_iter = it
        elif it - best_iter >= patience:
            print(f"[EarlyStop] best@{best_iter}, NLL={best:.3f}")
            break

    os.makedirs(args.run_dir, exist_ok=True)
    torch.save(theta.detach().cpu(), os.path.join(args.run_dir, "theta_raw.pt"))
    save_json(os.path.join(args.run_dir, "meta.json"), {
        "csv": args.csv, "dt": args.dt, "grid_n": args.grid_n,
        "J1": args.J1, "J2": args.J2, "R": args.R
    })

    run_filter(theta.detach(), y, args.dt, grid, args.J1, args.J2, args.R, args.run_dir, device, dtype)
    print("Saved run to:", args.run_dir)

if __name__ == "__main__":
    main()
