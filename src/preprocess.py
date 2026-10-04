"""Stage 2 - preprocess: scale, standardise, split train/val, save to data/processed/.

Run:  python src/preprocess.py
Reads hyperparameters from params.yaml -> preprocess: {val_size, seed, image_size}
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split

RAW = Path("data/raw")
OUT = Path("data/processed")


def load_raw(name):
    d = np.load(RAW / f"{name}.npz")
    return d["x"], d["y"]


def to_float(x, size):
    """uint8 NHWC -> float32 NCHW in [0, 1]; resized only if image_size != 32."""
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    if size != t.shape[-1]:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.numpy()


def normalize(x, mean, std):
    # >>> NORMALIZATION STEP (edited differently on both branches in Part E) <<<
    return (x - mean) / std


def main():
    p = yaml.safe_load(open("params.yaml"))["preprocess"]

    x, y = load_raw("train")
    x_test, y_test = load_raw("test")
    x, x_test = to_float(x, p["image_size"]), to_float(x_test, p["image_size"])

    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=p["val_size"], random_state=p["seed"], stratify=y
    )

    # per-channel statistics from the TRAINING split only (no leakage)
    mean = x_tr.mean(axis=(0, 2, 3), keepdims=True)
    std = x_tr.std(axis=(0, 2, 3), keepdims=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, xs, ys in (("train", x_tr, y_tr), ("val", x_val, y_val), ("test", x_test, y_test)):
        # float16 halves the storage / DVC push size; train.py casts back to float32
        np.savez_compressed(OUT / f"{name}.npz", x=normalize(xs, mean, std).astype(np.float16), y=ys)
        print(f"saved {name}: {xs.shape[0]} samples")

    with open(OUT / "stats.json", "w") as f:
        json.dump({"mean": mean.ravel().tolist(), "std": std.ravel().tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
