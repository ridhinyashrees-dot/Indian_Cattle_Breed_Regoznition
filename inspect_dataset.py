
import os
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def inspect_dataset(data_dir: Path):
    if not data_dir.exists():
        print(f"❌ ERROR: Directory '{data_dir}' not found!")
        return

    print("=" * 60)
    print(f"INSPECTING DATASET AT: {data_dir.resolve()}")
    print("=" * 60)

    # Automatically find all class directories containing images
    class_counts = {}
    corrupt_images = 0
    total_images = 0

    # Search recursively for directories containing image files
    all_subdirs = [d for d in data_dir.rglob("*") if d.is_dir()]

    for folder in sorted(all_subdirs):
        # Find images in current directory
        images = [
            f for f in folder.iterdir() 
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        if not images:
            continue

        # Use folder name as class label
        class_name = folder.name
        valid_count = 0

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_count += 1
            except Exception:
                corrupt_images += 1
                print(f"  [Corrupt File]: {img_path}")

        if valid_count > 0:
            class_counts[class_name] = valid_count
            total_images += valid_count

    if not class_counts:
        print("❌ No images found in dataset subfolders!")
        return

    print(f"\n{'Breed Class Name':<25} | {'Valid Image Count':<15}")
    print("-" * 45)
    for class_name, count in class_counts.items():
        print(f"{class_name:<25} | {count:<15}")

    print("-" * 45)
    print(f"Total Valid Images  : {total_images}")
    print(f"Total Corrupt Files : {corrupt_images}")
    print(f"Total Breed Classes : {len(class_counts)}")

if __name__ == "__main__":
    inspect_dataset(DATASET_DIR)