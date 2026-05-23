import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints")


def main():
    epochs, losses = [], []
    for f in sorted(os.listdir(CKPT_DIR)):
        if f.startswith("blip_epoch") and f.endswith(".pt"):
            ckpt = torch.load(os.path.join(CKPT_DIR, f), map_location="cpu")
            epochs.append(ckpt["epoch"])
            losses.append(ckpt["loss"])

    idx = np.argsort(epochs)
    epochs = np.array(epochs)[idx]
    losses = np.array(losses)[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, "b-o", markersize=8, linewidth=2, label="Train Loss")

    for e, l in zip(epochs, losses):
        ax.annotate(f"{l:.2f}", (e, l), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title("BLIP-2 Training Loss Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    fig.tight_layout()
    output_path = os.path.join(CKPT_DIR, "loss_curve.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved to {output_path}")
    print(f"Epochs: {epochs.tolist()}")
    print(f"Losses:  {[f'{l:.4f}' for l in losses]}")
    delta = losses[-1] - losses[0]
    print(f"Drop:    {delta:+.4f} ({losses[-1]/losses[0]*100:.1f}% of initial)")


if __name__ == "__main__":
    main()
