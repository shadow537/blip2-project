import os
import torch
import numpy as np
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from load_data import load_flickr8k_captions, load_images_and_captions, DATA_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = r"D:\blip2-main\clip-vit-model"
BATCH_SIZE = 16
OUTPUT_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ImageDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pixel_values = self.processor(images=item["image"], return_tensors="pt")["pixel_values"]
        return pixel_values.squeeze(0), item["filename"]


def main():
    print(f"Using device: {DEVICE}")

    # Load images using existing pipeline
    print("Loading captions and images...")
    image_captions = load_flickr8k_captions(
        os.path.join(DATA_DIR, "captions.txt"), max_images=200
    )
    data = load_images_and_captions(os.path.join(DATA_DIR, "Images"), image_captions)
    print(f"Loaded {len(data)} images.")

    # Load CLIP vision encoder and freeze it
    print(f"Loading CLIP vision encoder: {MODEL_NAME}")
    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Model frozen.")

    # Extract features
    dataset = ImageDataset(data, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pooled_dict = {}
    hidden_dict = {}

    print("Extracting features...")
    with torch.no_grad():
        for pixel_values, filenames in tqdm(dataloader):
            pixel_values = pixel_values.to(DEVICE)
            outputs = model(pixel_values, output_hidden_states=False)
            # last_hidden_state: (batch, 50, 768) — 1 CLS + 49 patch tokens
            # pooler_output:      (batch, 768)
            hidden = outputs.last_hidden_state.cpu().numpy()
            pooled = outputs.pooler_output.cpu().numpy()

            for fname, h, p in zip(filenames, hidden, pooled):
                hidden_dict[fname] = h
                pooled_dict[fname] = p

    # Save patch-level features for Q-Former
    filenames = sorted(hidden_dict.keys())
    hidden_matrix = np.stack([hidden_dict[f] for f in filenames])  # (200, 50, 768)
    pooled_matrix = np.stack([pooled_dict[f] for f in filenames])  # (200, 768)

    np.save(os.path.join(OUTPUT_DIR, "clip_vit_b32_filenames.npy"), np.array(filenames))
    np.save(os.path.join(OUTPUT_DIR, "clip_vit_b32_hidden.npy"), hidden_matrix)
    np.save(os.path.join(OUTPUT_DIR, "clip_vit_b32_pooled.npy"), pooled_matrix)
    print(f"Hidden states shape: {hidden_matrix.shape}")
    print(f"Pooled features shape: {pooled_matrix.shape}")


if __name__ == "__main__":
    main()
