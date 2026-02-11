from pathlib import Path
import json

data_dir = Path("data/processed")

# Load dataset info
with open(data_dir / "dataset_info.json") as f:
    info = json.load(f)
    print(json.dumps(info, indent=2))

# Count images per split/category
for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}:")
    split_dir = data_dir / split
    for cat_dir in sorted(split_dir.iterdir()):
        if cat_dir.is_dir():
            count = len(list(cat_dir.glob("*.jpg")))
            print(f"  {cat_dir.name}: {count} images")
