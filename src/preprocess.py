"""Stage 2 - preprocess: scale, standardize, split train/val, save to data/processed.

Run:
    python src/preprocess.py

Reads hyperparameters from params.yaml:
    preprocess:
        val_size
        seed
        image_size
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
    """Load raw images and labels from an NPZ file."""
    d = np.load(RAW / f"{name}.npz")
    return d["x"], d["y"]


def to_float(x, size):
    """
    Convert images from uint8 NHWC [0, 255]
    to float32 NCHW [0, 1].

    Resize images if image_size is not 32.
    """
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
    Per-image standardization.

    Each image is normalized independently using
    its own mean and standard deviation.

    Formula:
        normalized = (x - image_mean) / image_std
    """

    image_mean = x.mean(
        axis=(1, 2, 3),
        keepdims=True
    )

    image_std = x.std(
        axis=(1, 2, 3),
        keepdims=True
    )

    # Small epsilon prevents division by zero
    return (x - image_mean) / (image_std + 1e-8)


def main():

    # ---------------------------------------------------------
    # 1. Read preprocessing parameters
    # ---------------------------------------------------------
    with open("params.yaml", "r") as f:
        p = yaml.safe_load(f)["preprocess"]

    # ---------------------------------------------------------
    # 2. Load raw training and test data
    # ---------------------------------------------------------
    x, y = load_raw("train")
    x_test, y_test = load_raw("test")

    # ---------------------------------------------------------
    # 3. Convert images:
    #    uint8 [0,255] -> float32 [0,1]
    #    NHWC -> NCHW
    # ---------------------------------------------------------
    x = to_float(x, p["image_size"])
    x_test = to_float(x_test, p["image_size"])

    # ---------------------------------------------------------
    # 4. Split training data into training and validation sets
    # ---------------------------------------------------------
    x_tr, x_val, y_tr, y_val = train_test_split(
        x,
        y,
        test_size=p["val_size"],
        random_state=p["seed"],
        stratify=y
    )

    # ---------------------------------------------------------
    # 5. Create output directory
    # ---------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 6. Normalize and save datasets
    # ---------------------------------------------------------
    datasets = (
        ("train", x_tr, y_tr),
        ("val", x_val, y_val),
        ("test", x_test, y_test)
    )

    for name, xs, ys in datasets:

        # Apply per-image standardization
        normalized_x = normalize(xs)

        # float16 reduces storage size
        np.savez_compressed(
            OUT / f"{name}.npz",
            x=normalized_x.astype(np.float16),
            y=ys
        )

        print(
            f"saved {name}: "
            f"{xs.shape[0]} samples, "
            f"shape={normalized_x.shape}"
        )

    # ---------------------------------------------------------
    # 7. Save normalization information
    # ---------------------------------------------------------
    stats = {
        "normalization": "per_image_standardization",
        "formula": "(x - image_mean) / (image_std + 1e-8)",
        "epsilon": 1e-8
    }

    # Save normalization statistics
    with open(OUT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()