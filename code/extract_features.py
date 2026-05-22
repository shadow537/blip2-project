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

    features_dict = {}

    print("Extracting features...")
    with torch.no_grad():
        for pixel_values, filenames in tqdm(dataloader):
            pixel_values = pixel_values.to(DEVICE)
            outputs = model(pixel_values)
            # CLIPVisionModel outputs pooler_output: (batch, 768)
            features = outputs.pooler_output.cpu().numpy()

            for fname, feat in zip(filenames, features):
                features_dict[fname] = feat

    # Save features
    output_path = os.path.join(OUTPUT_DIR, "clip_vit_b32_features.npz")
    np.savez(output_path, **features_dict)
    print(f"Saved {len(features_dict)} feature vectors to {output_path}")

    # Also save filenames list and feature matrix for convenience
    filenames = sorted(features_dict.keys())
    feature_matrix = np.stack([features_dict[f] for f in filenames])
    np.save(os.path.join(OUTPUT_DIR, "clip_vit_b32_filenames.npy"), np.array(filenames))
    np.save(os.path.join(OUTPUT_DIR, "clip_vit_b32_features.npy"), feature_matrix)
    print(f"Feature matrix shape: {feature_matrix.shape}")


if __name__ == "__main__":
    main()
