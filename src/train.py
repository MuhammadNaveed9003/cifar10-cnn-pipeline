"""Stage 3 - train: build and train the CNN, save models/model.pth and models/history.csv.

Run:  python src/train.py
Reads hyperparameters from params.yaml -> train: {...} and preprocess.image_size
"""
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

DATA = Path("data/processed")
OUT = Path("models")


def build_model(num_filters, dropout_rate, dense_units, image_size):
    """Conv-BN-ReLU-MaxPool -> Conv-BN-ReLU-MaxPool -> Dense -> Dropout -> Output(10)."""
    s = image_size // 4  # two 2x2 max-pools
    return nn.Sequential(
        nn.Conv2d(3, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(num_filters, num_filters * 2, 3, padding=1), nn.BatchNorm2d(num_filters * 2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(num_filters * 2 * s * s, dense_units), nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, 10),
    )


def load(name):
    d = np.load(DATA / f"{name}.npz")
    return torch.from_numpy(d["x"].astype(np.float32)), torch.from_numpy(d["y"])


def run_epoch(model, x, y, p, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    idx = torch.randperm(len(x)) if training else torch.arange(len(x))
    loss_sum, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for i in range(0, len(x), p["batch_size"]):
            j = idx[i:i + p["batch_size"]]
            xb, yb = x[j].to(device), y[j].to(device)
            if training and p["augment_flip"]:  # random horizontal flip
                flip = torch.rand(len(xb), device=device) < 0.5
                xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            out = model(xb)
            loss = criterion(out, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
    return loss_sum / len(x), correct / len(x)


def main():
    params = yaml.safe_load(open("params.yaml"))
    p = params["train"]

    random.seed(p["seed"]); np.random.seed(p["seed"]); torch.manual_seed(p["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    x_tr, y_tr = load("train")
    x_val, y_val = load("val")

    model = build_model(p["num_filters"], p["dropout_rate"], p["dense_units"],
                        params["preprocess"]["image_size"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])

    OUT.mkdir(exist_ok=True)
    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, p["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, x_tr, y_tr, p, device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, x_val, y_val, p, device, criterion)
        history.append([epoch, tr_loss, tr_acc, va_loss, va_acc])
        print(f"epoch {epoch:02d}/{p['epochs']}  train {tr_loss:.4f}/{tr_acc:.4f}  val {va_loss:.4f}/{va_acc:.4f}")
        if va_acc > best_acc:  # keep the best-validation checkpoint
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, OUT / "model.pth")
    with open(OUT / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(history)
    print(f"saved {OUT / 'model.pth'} (best val acc {best_acc:.4f}) and {OUT / 'history.csv'}")


if __name__ == "__main__":
    main()
