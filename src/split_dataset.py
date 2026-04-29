import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Takes a flat class-folder dataset and creates a train/test split.
    source_dir/
        pose_a/  img1.jpg  img2.jpg ...
        pose_b/  ...
    becomes:
    output_dir/
        train/  pose_a/  pose_b/  ...
        test/   pose_a/  pose_b/  ...
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    classes = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes")

    for cls_dir in classes:
        images = list(cls_dir.glob("*.jpg")) + \
                 list(cls_dir.glob("*.jpeg")) + \
                 list(cls_dir.glob("*.png"))

        if len(images) == 0:
            print(f"  Skipping {cls_dir.name} — no images found")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs  = images[split_idx:]

        for split, imgs in [("train", train_imgs), ("test", test_imgs)]:
            dest = output_dir / split / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  {cls_dir.name}: {len(train_imgs)} train | {len(test_imgs)} test")

    print(f"\nDone. Split dataset saved to {output_dir}")

if __name__ == "__main__":
    # Point this at wherever you unzipped the Kaggle download
    split_dataset(
        source_dir="data/raw/dataset",   
        output_dir="data/raw",
        train_ratio=0.8
    )