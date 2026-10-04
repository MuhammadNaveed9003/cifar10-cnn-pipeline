"""Stage 1 - prepare: download CIFAR-10 and save raw uint8 arrays to data/raw/.

Run:  python src/prepare.py
No hyperparameters are needed at this stage.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

OUT = Path("data/raw")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()  # torchvision's download cache is thrown away afterwards
    try:
        for name, is_train in (("train", True), ("test", False)):
            ds = CIFAR10(root=tmp, train=is_train, download=True)
            x = ds.data                                  # (N, 32, 32, 3) uint8
            y = np.array(ds.targets, dtype=np.int64)     # (N,)
            np.savez_compressed(OUT / f"{name}.npz", x=x, y=y)
            print(f"saved {name}: x={x.shape} y={y.shape} -> {OUT / (name + '.npz')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
