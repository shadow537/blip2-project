"""Caption generation with beam search on 5 held-out test images.

Ground-truth captions are shown alongside the model's generated captions
for qualitative comparison.
"""

import os
import sys
import numpy as np
import torch

from load_data import load_flickr8k_captions, DATA_DIR
from blip_model import BLIPModel


FEATURES_DIR = os.path.join(DATA_DIR, "features")
HIDDEN_PATH = os.path.join(FEATURES_DIR, "clip_vit_b32_hidden.npy")
FILENAMES_PATH = os.path.join(FEATURES_DIR, "clip_vit_b32_filenames.npy")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

NUM_TEST = 5
NUM_BEAMS = 5


def load_checkpoint(model, checkpoint_path):
    """Load a saved model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=model.device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        info = f"epoch {ckpt.get('epoch', '?')}, avg_loss={ckpt.get('loss', '?'):.4f}"
    else:
        model.load_state_dict(ckpt)
        info = "final"
    return info


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load data ----
    all_hidden = np.load(HIDDEN_PATH)         # (200, 50, 768)
    all_fnames = np.load(FILENAMES_PATH)       # (200,)
    image_captions = load_flickr8k_captions(CAPTION_FILE, max_images=200)

    # The last 5 images are held out for testing
    test_fnames = list(all_fnames[-NUM_TEST:])
    fname_to_idx = {fname: i for i, fname in enumerate(all_fnames)}

    # ---- Load model ----
    print("Loading BLIP model...")
    model = BLIPModel(
        opt_path=r"D:\blip2-main\opt-125m",
        vision_dim=768, hidden_dim=768,
        num_queries=32, num_qformer_layers=2, num_heads=8,
        dropout=0.1, device=device,
    )
    model.to(device)
    model.eval()

    # Try loading a trained checkpoint; fall back to untrained
    candidate_ckpts = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        reverse=True,
    ) if os.path.isdir(CHECKPOINT_DIR) else []

    if candidate_ckpts:
        ckpt_path = os.path.join(CHECKPOINT_DIR, candidate_ckpts[0])
        info = load_checkpoint(model, ckpt_path)
        print(f"Loaded checkpoint: {candidate_ckpts[0]} ({info})")
    else:
        print("No checkpoint found — using untrained model (output will be random).")

    # ---- Generate & compare ----
    print(f"\n{'='*80}")
    print(f"Beam-search caption generation (beams={NUM_BEAMS}) on {NUM_TEST} test images")
    print(f"{'='*80}")

    for i, fname in enumerate(test_fnames):
        idx = fname_to_idx[fname]
        vision_feat = torch.from_numpy(all_hidden[idx]).float().unsqueeze(0).to(device)

        # Ground-truth captions
        gt_captions = image_captions.get(fname, ["(no captions found)"])

        # Generate with beam search (top-1 best scoring candidate)
        with torch.no_grad():
            generated = model.generate_beam(
                vision_feat,
                prompt=None,
                num_beams=NUM_BEAMS,
                max_new_tokens=64,
                num_return=1,
            )

        print(f"\n--- Image {i+1}/{NUM_TEST}: {fname} ---")
        print(f"Ground truth ({len(gt_captions)} captions):")
        for j, cap in enumerate(gt_captions, 1):
            print(f"  [{j}] {cap}")

        print(f"\nGenerated (beam={NUM_BEAMS}):")
        print(f"  {generated[0]}")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
