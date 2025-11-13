import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_modes(run_dir):
    with open(os.path.join(run_dir, "K_list.json"), "r") as f:
        data = json.load(f)
    return [tuple(x) for x in data]

def make_time_labels(T, start_str="11:07", step_min=20):
    hh, mm = map(int, start_str.split(":"))
    t0 = datetime(2000, 1, 1, hh, mm)
    times = [t0 + timedelta(minutes=step_min)*i for i in range(T)]
    labels = [(t.strftime("%I:%M %p")).lstrip("0") for t in times]
    return labels, times

def time_label_for_index(idx, start_str="11:07", step_min=20):
    labels, _ = make_time_labels(idx+1, start_str, step_min)
    return labels[idx]

def _ticks_every_3_with_last(T):
    return np.unique(np.r_[np.arange(0, T, 3), T-1])

def plot_mag(ts, title, save_path, vline_idx=None, start_str="11:07", step_min=20):
    T = len(ts)
    x = np.arange(T)
    xtick_labels, _ = make_time_labels(T, start_str, step_min)
    ticks = _ticks_every_3_with_last(T)

    plt.figure(figsize=(4.8, 3.0), dpi=200)
    plt.plot(x, ts, linewidth=1.2)
    if vline_idx is not None and 0 <= vline_idx < T:
        vlabel = time_label_for_index(vline_idx, start_str, step_min)
        plt.axvline(vline_idx, color='red', linestyle='--', linewidth=1.0, label=vlabel)
        plt.legend(loc='best', frameon=False, fontsize=9)
    plt.xlabel("time")
    plt.title(title)
    plt.xticks(ticks, [xtick_labels[i] for i in ticks], rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_two(ts1, ts2, title1, title2, save_path, vline_idx=None, start_str="11:07", step_min=20):
    T1, T2 = len(ts1), len(ts2)
    x1, x2 = np.arange(T1), np.arange(T2)
    labels1, _ = make_time_labels(T1, start_str, step_min)
    labels2, _ = make_time_labels(T2, start_str, step_min)
    ticks1 = _ticks_every_3_with_last(T1)
    ticks2 = _ticks_every_3_with_last(T2)

    fig = plt.figure(figsize=(8.0, 3.0), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x1, ts1, linewidth=1.0)
    if vline_idx is not None and 0 <= vline_idx < T1:
        vlabel = time_label_for_index(vline_idx, start_str, step_min)
        ax1.axvline(vline_idx, color='red', linestyle='--', linewidth=1.0, label=vlabel)
        ax1.legend(loc='best', frameon=False, fontsize=8)
    ax1.set_title(title1); ax1.set_xlabel("time")
    ax1.set_xticks(ticks1); ax1.set_xticklabels([labels1[i] for i in ticks1], rotation=35, ha="right")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x2, ts2, linewidth=1.0)
    if vline_idx is not None and 0 <= vline_idx < T2:
        vlabel = time_label_for_index(vline_idx, start_str, step_min)
        ax2.axvline(vline_idx, color='red', linestyle='--', linewidth=1.0, label=vlabel)
        ax2.legend(loc='best', frameon=False, fontsize=8)
    ax2.set_title(title2); ax2.set_xlabel("time")
    ax2.set_xticks(ticks2); ax2.set_xticklabels([labels2[i] for i in ticks2], rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def main(alpha_csv, beta_csv, run_dir, out_dir, vline_idx=9, start_str="11:07", step_min=20):
    os.makedirs(out_dir, exist_ok=True)
    for sub in ["alpha_mag", "beta_mag", "alpha_comp", "beta_comp"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    alpha = np.loadtxt(alpha_csv, delimiter=",")
    beta  = np.loadtxt(beta_csv,  delimiter=",")
    T, twoN = alpha.shape
    if beta.shape != (T, twoN):
        raise ValueError("alpha/beta shape mismatch")

    n_modes = twoN // 2
    K_list = load_modes(run_dir)
    if len(K_list) != n_modes:
        raise ValueError(f"modes mismatch: K_list={len(K_list)} vs CSV={n_modes}")

    # save mode index map
    with open(os.path.join(out_dir, "modes_index.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode_idx", "k1", "k2"])
        for idx, (i, j) in enumerate(K_list):
            writer.writerow([idx, i, j])

    # plot all modes
    for j, (k1_val, k2_val) in enumerate(K_list):
        a_cos = alpha[:, 2*j]
        a_sin = alpha[:, 2*j + 1]
        b_cos = beta[:, 2*j]
        b_sin = beta[:, 2*j + 1]

        a_mag = np.sqrt(a_cos**2 + a_sin**2)
        b_mag = np.sqrt(b_cos**2 + b_sin**2)
        tag = f"{k1_val}_{k2_val}"

        plot_mag(
            a_mag,
            f"|alpha|  (k1,k2)=({k1_val},{k2_val})",
            os.path.join(out_dir, "alpha_mag", f"alpha_mag_{tag}.png"),
            vline_idx=vline_idx, start_str=start_str, step_min=step_min
        )
        plot_mag(
            b_mag,
            f"|beta|   (k1,k2)=({k1_val},{k2_val})",
            os.path.join(out_dir, "beta_mag", f"beta_mag_{tag}.png"),
            vline_idx=vline_idx, start_str=start_str, step_min=step_min
        )
        plot_two(
            a_cos, a_sin,
            f"alpha_cos  (k1,k2)=({k1_val},{k2_val})",
            f"alpha_sin  (k1,k2)=({k1_val},{k2_val})",
            os.path.join(out_dir, "alpha_comp", f"alpha_comp_{tag}.png"),
            vline_idx=vline_idx, start_str=start_str, step_min=step_min
        )
        plot_two(
            b_cos, b_sin,
            f"beta_cos   (k1,k2)=({k1_val},{k2_val})",
            f"beta_sin   (k1,k2)=({k1_val},{k2_val})",
            os.path.join(out_dir, "beta_comp", f"beta_comp_{tag}.png"),
            vline_idx=vline_idx, start_str=start_str, step_min=step_min
        )

    print(f"[done] wrote plots to: {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha_csv", required=True)
    ap.add_argument("--beta_csv", required=True)
    ap.add_argument("--run_dir", required=True, help="directory containing K_list.json")
    ap.add_argument("--out_dir", default="plots_alpha_beta")
    ap.add_argument("--vline_idx", type=int, default=9)
    ap.add_argument("--start_time", default="11:07", help="start clock label, e.g. 11:07")
    ap.add_argument("--step_min", type=int, default=20, help="minutes per step for tick labels")
    args = ap.parse_args()
    main(args.alpha_csv, args.beta_csv, args.run_dir, args.out_dir, args.vline_idx, args.start_time, args.step_min)
