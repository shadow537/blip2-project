import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from load_data import load_flickr8k_captions, DATA_DIR
from blip_model import BLIPModel


FEATURES_DIR = os.path.join(DATA_DIR, "features")
HIDDEN_PATH = os.path.join(FEATURES_DIR, "clip_vit_b32_hidden.npy")
FILENAMES_PATH = os.path.join(FEATURES_DIR, "clip_vit_b32_filenames.npy")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class BLIPDataset(Dataset):
    """Pairs pre-extracted vision features with captions.

    Args:
        exclude_filenames: optional set of filenames to exclude (e.g. held-out test set).
    """

    def __init__(self, hidden_path, filenames_path, caption_file, tokenizer,
                 max_images=200, exclude_filenames=None):
        self.hidden_states = np.load(hidden_path)       # (N, 50, 768)
        self.filenames = np.load(filenames_path)         # (N,)
        self.tokenizer = tokenizer

        # Build filename → index mapping
        self.filename_to_idx = {fname: i for i, fname in enumerate(self.filenames)}

        # Load captions
        image_captions = load_flickr8k_captions(caption_file, max_images=max_images)

        # Build samples: each (image, caption) is one sample
        exclude = exclude_filenames or set()
        num_skipped = 0
        self.samples = []
        for fname, captions in image_captions.items():
            if fname not in self.filename_to_idx:
                continue
            if fname in exclude:
                num_skipped += 1
                continue
            for cap in captions:
                self.samples.append((fname, cap))

        if num_skipped:
            print(f"  Excluded {num_skipped} images (held-out test set), "
                  f"{len(self.samples)} samples remaining.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        feat_idx = self.filename_to_idx[fname]
        vision_features = torch.from_numpy(self.hidden_states[feat_idx]).float()

        enc = self.tokenizer(
            caption,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=64,
        )
        input_ids = enc["input_ids"].squeeze(0)  # (seq_len,)

        return {
            "vision_features": vision_features,  # (50, 768)
            "input_ids": input_ids,               # (seq_len,)
        }


def collate_fn(batch, tokenizer):
    """Pad input_ids to same length within a batch."""
    vision_features = torch.stack([item["vision_features"] for item in batch])

    input_ids_list = [item["input_ids"] for item in batch]
    input_ids_padded = nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    return {
        "vision_features": vision_features,
        "input_ids": input_ids_padded,
    }


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # Initialize model
    print("Initializing BLIP model...")
    model = BLIPModel(
        opt_path=r"D:\blip2-main\opt-125m",
        vision_dim=768,
        hidden_dim=768,
        num_queries=32,
        num_qformer_layers=2,
        num_heads=8,
        dropout=0.1,
        device=device,
    )
    model.to(device)

    # Print trainable params summary
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_trainable:,}")
    print(f"  - MiniQFormer:           {sum(p.numel() for n, p in model.qformer.named_parameters()):,}")

    tokenizer = model.opt_decoder.tokenizer

    # Dataset & DataLoader — hold out last 5 images for testing
    all_fnames = np.load(FILENAMES_PATH)
    test_fnames = set(all_fnames[-5:].tolist())
    print(f"Held-out test images: {sorted(test_fnames)}")

    print("Loading dataset...")
    dataset = BLIPDataset(
        HIDDEN_PATH, FILENAMES_PATH, CAPTION_FILE, tokenizer,
        max_images=200, exclude_filenames=test_fnames,
    )
    print(f"Total samples: {len(dataset)} (train images × ~5 captions)")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=0,
    )

    # Optimizer & scheduler
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(dataloader) * NUM_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    print(f"\nStarting training: {NUM_EPOCHS} epochs, {len(dataloader)} batches/epoch")
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for batch in pbar:
            vision_features = batch["vision_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()

            optimizer.zero_grad()

            out = model(vision_features, input_ids, labels)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} — avg loss: {avg_loss:.4f}")

        # Save checkpoint every 2 epochs
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"blip_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
         }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "blip_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    return model


if __name__ == "__main__":
    train()
