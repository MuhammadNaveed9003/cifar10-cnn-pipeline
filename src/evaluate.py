"""Stage 4 - evaluate: test loss/accuracy -> metrics.json, confusion matrix -> plots/confusion_matrix.png.

Run:  python src/evaluate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn

from train import build_model, load  # reuse the exact architecture used for training

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    params = yaml.safe_load(open("params.yaml"))
    t = params["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(t["num_filters"], t["dropout_rate"], t["dense_units"],
                        params["preprocess"]["image_size"]).to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.eval()

    x, y = load("test")
    criterion = nn.CrossEntropyLoss()
    loss_sum, preds = 0.0, []
    with torch.no_grad():
        for i in range(0, len(x), 500):
            xb, yb = x[i:i + 500].to(device), y[i:i + 500].to(device)
            out = model(xb)
            loss_sum += criterion(out, yb).item() * len(xb)
            preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    y = y.numpy()

    metrics = {
        "test_loss": round(loss_sum / len(x), 4),
        "test_accuracy": round(float((preds == y).mean()), 4),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

    Path("plots").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix(y, preds), display_labels=CLASSES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"CIFAR-10 confusion matrix (test acc {metrics['test_accuracy']:.2%})")
    fig.tight_layout()
    fig.savefig("plots/confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
