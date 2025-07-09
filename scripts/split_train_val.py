import os
import shutil
import random

# Set these paths to your data folders
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
VAL_RATIO = 0.4  # 40% for validation

random.seed(42)

os.makedirs(VAL_DIR, exist_ok=True)

for class_name in os.listdir(TRAIN_DIR):
    class_train_dir = os.path.join(TRAIN_DIR, class_name)
    class_val_dir = os.path.join(VAL_DIR, class_name)
    if not os.path.isdir(class_train_dir):
        continue
    os.makedirs(class_val_dir, exist_ok=True)
    images = [f for f in os.listdir(class_train_dir) if os.path.isfile(os.path.join(class_train_dir, f))]
    random.shuffle(images)
    n_val = int(len(images) * VAL_RATIO)
    val_images = images[:n_val]
    for img in val_images:
        src = os.path.join(class_train_dir, img)
        dst = os.path.join(class_val_dir, img)
        shutil.move(src, dst)
    print(f"Moved {len(val_images)} images from {class_train_dir} to {class_val_dir}")

print("Data split complete.")
