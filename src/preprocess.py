"""Stage 2 - preprocess: scale, normalize, split train/val, save to data/processed/.

Run: python src/preprocess.py
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
    """uint8 NHWC -> float32 NCHW in [0, 1]; resized if required."""
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0

    if size != t.shape[-1]:
        t = F.interpolate(
            t,
            size=(size, size),
            mode="bilinear",
            align_corners=False
        )

    return t.numpy()


def normalize(x):
    """
    Normalize images using standard CIFAR-10
    channel-wise mean and standard deviation.
    """

    mean = np.array(
        [0.4914, 0.4822, 0.4465],
        dtype=np.float32
    ).reshape(1, 3, 1, 1)

    std = np.array(
        [0.2470, 0.2435, 0.2616],
        dtype=np.float32
    ).reshape(1, 3, 1, 1)

    return (x - mean) / std


def main():

    p = yaml.safe_load(open("params.yaml"))["preprocess"]

    # Load raw data
    x, y = load_raw("train")
    x_test, y_test = load_raw("test")

    # Convert uint8 [0,255] -> float32 [0,1]
    x = to_float(x, p["image_size"])
    x_test = to_float(x_test, p["image_size"])

    # Split training data into train and validation
    x_tr, x_val, y_tr, y_val = train_test_split(
        x,
        y,
        test_size=p["val_size"],
        random_state=p["seed"],
        stratify=y
    )

    OUT.mkdir(parents=True, exist_ok=True)

    # Normalize and save datasets
    for name, xs, ys in (
        ("train", x_tr, y_tr),
        ("val", x_val, y_val),
        ("test", x_test, y_test)
    ):

        normalized_x = normalize(xs)

        np.savez_compressed(
            OUT / f"{name}.npz",
            x=normalized_x.astype(np.float16),
            y=ys
        )

        print(f"saved {name}: {xs.shape[0]} samples")

    # Save normalization statistics
    with open(OUT / "stats.json", "w") as f:
        json.dump(
            {
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616]
            },
            f,
            indent=2
        )


if __name__ == "__main__":
    main()