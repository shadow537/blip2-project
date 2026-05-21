import os
from PIL import Image
from collections import defaultdict

DATA_DIR = r"D:\blip2-main\data"
IMAGE_DIR = os.path.join(DATA_DIR, "Images")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")


def load_flickr8k_captions(caption_file, max_images=200):
    """Load captions for the first `max_images` unique images."""
    image_captions = defaultdict(list)

    with open(caption_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            img_name, caption = parts
            caption = caption.strip().strip('"')
            if len(image_captions) >= max_images and img_name not in image_captions:
                break
            image_captions[img_name].append(caption)

    return dict(image_captions)


def load_images_and_captions(image_dir, image_captions):
    """Load PIL Images for each image that has captions."""
    data = []
    for img_name, captions in image_captions.items():
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_name} not found, skipping.")
            continue
        image = Image.open(img_path).convert("RGB")
        data.append({"image": image, "captions": captions, "filename": img_name})
    return data


if __name__ == "__main__":
    print("Loading captions...")
    image_captions = load_flickr8k_captions(CAPTION_FILE, max_images=200)
    print(f"Loaded captions for {len(image_captions)} images.")

    print("Loading images...")
    data = load_images_and_captions(IMAGE_DIR, image_captions)
    print(f"Loaded {len(data)} images with captions.")

    # Quick preview
    for item in data[:3]:
        print(f"\n{item['filename']} ({item['image'].size}):")
        for cap in item["captions"]:
            print(f"  - {cap}")
