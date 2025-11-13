import os, json
import pandas as pd
import torch

def load_csv_as_tensor(path, device, dtype):
    df = pd.read_csv(path, index_col=0)
    return torch.tensor(df.values, device=device, dtype=dtype)

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_run_arrays(run_dir, arrays: dict):
    os.makedirs(run_dir, exist_ok=True)
    for name, arr in arrays.items():
        torch.save(arr, os.path.join(run_dir, f"{name}.pt"))
