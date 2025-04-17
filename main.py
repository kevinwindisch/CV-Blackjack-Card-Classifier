# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

import kagglehub
from kagglehub import KaggleDatasetAdapter
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()





import os
import shutil
import random
import kagglehub

# === Step 1: Download Dataset ===
dataset_path = kagglehub.dataset_download("jaypradipshah/the-complete-playing-card-dataset")
print("✅ Downloaded dataset to:", dataset_path)

# === Step 2: Prepare YOLOv8 folder structure ===
target_base = "data/cards"
paths = {
    "images/train": os.path.join(target_base, "images/train"),
    "images/val": os.path.join(target_base, "images/val"),
    "labels/train": os.path.join(target_base, "labels/train"),
    "labels/val": os.path.join(target_base, "labels/val"),
}
for p in paths.values():
    os.makedirs(p, exist_ok=True)

# === Step 3: Find and collect all image/label files ===
image_exts = (".jpg", ".jpeg", ".png")
all_images = []
all_labels = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(image_exts):
            all_images.append(os.path.join(root, file))
        elif file.endswith(".txt"):
            all_labels.append(os.path.join(root, file))

# === Step 4: Match labels to images ===
paired = []
for img_path in all_images:
    fname = os.path.splitext(os.path.basename(img_path))[0]
    matching_label = [l for l in all_labels if os.path.splitext(os.path.basename(l))[0] == fname]
    if matching_label:
        paired.append((img_path, matching_label[0]))

print(f"✅ Found {len(paired)} image-label pairs")

# === Step 5: Train/Val Split (80/20) ===
random.shuffle(paired)
split_idx = int(0.8 * len(paired))
train_pairs = paired[:split_idx]
val_pairs = paired[split_idx:]

def move_pairs(pairs, img_dest, lbl_dest):
    for img, lbl in pairs:
        shutil.copy2(img, os.path.join(img_dest, os.path.basename(img)))
        shutil.copy2(lbl, os.path.join(lbl_dest, os.path.basename(lbl)))

move_pairs(train_pairs, paths["images/train"], paths["labels/train"])
move_pairs(val_pairs, paths["images/val"], paths["labels/val"])

print(f"✅ {len(train_pairs)} training and {len(val_pairs)} validation samples prepared.")

# === Step 6: Create data.yaml ===
yaml_content = """\
path: data/cards
train: images/train
val: images/val

nc: 52
names: [
  "2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S",
  "4C", "4D", "4H", "4S", "5C", "5D", "5H", "5S",
  "6C", "6D", "6H", "6S", "7C", "7D", "7H", "7S",
  "8C", "8D", "8H", "8S", "9C", "9D", "9H", "9S",
  "10C", "10D", "10H", "10S", "JC", "JD", "JH", "JS",
  "QC", "QD", "QH", "QS", "KC", "KD", "KH", "KS",
  "AC", "AD", "AH", "AS"
]
"""

with open(os.path.join(target_base, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("✅ data.yaml created and full dataset is ready for training.")



# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # Load the base model
# model.train(data='data/cards/data.yaml', epochs=100, imgsz=640)