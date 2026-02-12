# Asset Download Guide

This document provides detailed instructions for downloading the required assets that are not included in the Git repository.

---

## 📦 Required Assets Overview

| Asset | Size | Purpose | Required |
|-------|------|---------|----------|
| **Dataset** | ~750 MB | Training/validation/test images | ✅ Essential |
| **Trained Model** | ~50 MB | Pre-trained EfficientNet-B0 checkpoint | ✅ Essential |

---

## 1️⃣ Dataset Download

### Option A: Google Drive (Recommended)

**Download Link:** [https://drive.google.com/drive/folders/1g1V4I3WL8FfXLZfkXqrTYcCR8etojiBY?usp=drive_link](https://drive.google.com/drive/folders/1g1V4I3WL8FfXLZfkXqrTYcCR8etojiBY?usp=drive_link)

**Steps:**
1. Click the download link above
2. Download `dataset.zip` (~750 MB)
3. Extract to your project directory:
   ```bash
   unzip dataset.zip -d data/
   ```

**Verify:**
```bash
tree data/processed -L 2

# Expected output:
# data/processed/
# ├── dataset_info.json
# ├── test/
# │   ├── casual shoes/
# │   ├── handbags/
# │   ├── shirts/
# │   ├── tops/
# │   └── watches/
# ├── train/
# │   └── (same 5 categories)
# └── val/
#     └── (same 5 categories)

# Check image counts
find data/processed/train -name "*.jpg" | wc -l  # Should be ~1,750
find data/processed/val -name "*.jpg" | wc -l    # Should be ~375
find data/processed/test -name "*.jpg" | wc -l   # Should be ~375
```

### Option B: Kaggle (Recreate from Source)

If you prefer to recreate the dataset from the original source:

**Source:** [Fashion Product Images Dataset on Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

**Steps:**
1. Download the full Kaggle dataset
2. Open `notebooks/01_data_preparation.ipynb` in Kaggle or locally
3. Run all cells to sample and split the data
4. Download the output to `data/processed/`

**Note:** This ensures reproducibility but requires ~1GB download and processing time.

---

## 2️⃣ Trained Model Download

### Option A: Google Drive (Recommended)

**Download Link:** [https://drive.google.com/drive/folders/1IQ4wyuTYO0TuQvKg0bIZ3n1kTZQpuvtp?usp=drive_link](https://drive.google.com/drive/folders/1IQ4wyuTYO0TuQvKg0bIZ3n1kTZQpuvtp?usp=drive_link)

**Steps:**
1. Click the download link above
2. Download `model_v1.zip` (~20 MB)
3. Extract to your project directory:
   ```bash
   unzip model_v1.zip -d models/
   ```

**Verify:**
```bash
ls -lh models/v1/

# Expected output:
# best_model.pth           (~20 MB)  - PyTorch model checkpoint
# training_metadata.json   (~5 KB)   - Hyperparameters and metrics
```

**Check model metadata:**
```bash
cat models/v1/training_metadata.json | jq
```

Should show:
- `model_architecture`: "efficientnet_b0"
- `test_acc`: 96.53
- `num_classes`: 5
- Training hyperparameters

### Option B: Retrain from Scratch

If you want to train your own model:

**Prerequisites:**
- Google Colab account (for free GPU)
- Dataset downloaded (see Option A above)

**Steps:**
1. Upload dataset to Google Drive
2. Open `notebooks/02_model_training.ipynb` in Google Colab
3. Mount your Google Drive
4. Run all cells (~12 minutes on T4 GPU)
5. Download generated files to `models/v1/`:
   - `best_model.pth`
   - `training_metadata.json`

**Note:** You may get slightly different accuracy due to randomness, but should be within ±1% of 96.53%.

---


## 🔍 Verification Checklist

After downloading all assets, run this verification:

```bash
# Option 1: Use automated script
chmod +x scripts/setup_assets.sh
./scripts/setup_assets.sh

# Option 2: Manual verification
echo "Checking dataset..."
[ -d "data/processed/train" ] && echo "✓ Dataset found" || echo "✗ Dataset missing"

echo "Checking model..."
[ -f "models/v1/best_model.pth" ] && echo "✓ Model found" || echo "✗ Model missing"

echo "Counting images..."
echo "Train: $(find data/processed/train -name '*.jpg' | wc -l) (expected: 1,750)"
echo "Val:   $(find data/processed/val -name '*.jpg' | wc -l) (expected: 375)"
echo "Test:  $(find data/processed/test -name '*.jpg' | wc -l) (expected: 375)"
```

**All checks should pass before proceeding to Quick Start.**

---

## 📊 Dataset Details

### Categories (5 total)

| Category | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| Casual Shoes | 350 | 75 | 75 | 500 |
| Handbags | 350 | 75 | 75 | 500 |
| Shirts | 350 | 75 | 75 | 500 |
| Tops | 350 | 75 | 75 | 500 |
| Watches | 350 | 75 | 75 | 500 |
| **Total** | **1,750** | **375** | **375** | **2,500** |

### Image Properties

- **Format:** JPEG
- **Size:** Variable (original product photos)
- **Preprocessing:** Images are resized to 224×224 during model training
- **Channels:** RGB (3 channels)
- **Source:** Fashion Product Images dataset (Kaggle)

---

## 🧪 Model Details

### Architecture: EfficientNet-B0

**Specifications:**
- **Parameters:** ~5.3 million
- **Input Size:** 224×224×3
- **Output:** 5 classes (softmax probabilities)
- **Pretrained:** ImageNet weights
- **Fine-tuning:** All layers trained (not just classifier)

### Training Configuration

```json
{
  "model_architecture": "efficientnet_b0",
  "num_classes": 5,
  "img_size": 224,
  "batch_size": 32,
  "num_epochs": 15,
  "learning_rate": 0.0001,
  "optimizer": "Adam",
  "scheduler": "ReduceLROnPlateau",
  "data_augmentation": [
    "RandomHorizontalFlip",
    "RandomRotation(15)",
    "ColorJitter(0.2, 0.2)"
  ]
}
```

### Performance Metrics

- **Test Accuracy:** 96.53%
- **Validation Accuracy:** 98.40%
- **Training Time:** ~12 minutes (Google Colab T4 GPU)
- **Inference Speed:** ~100-200ms per image (CPU)
- **Model Size:** 20 MB (checkpoint file)

---

## 🆘 Troubleshooting

### "Dataset structure doesn't match expected"

**Issue:** Folder structure is wrong after extraction

**Solution:**
```bash
# Correct structure
data/processed/
  ├── train/
  │   ├── casual shoes/
  │   ├── handbags/
  │   └── ...

# If you have an extra nested folder:
mv data/processed/fashion_subset/* data/processed/
rmdir data/processed/fashion_subset
```

### "Model file is corrupted"

**Issue:** Download interrupted or file damaged

**Solution:**
1. Delete partial download
2. Re-download from link
3. Verify file size: `ls -lh models/v1/best_model.pth` should show ~20MB
4. Check hash (if provided): `sha256sum models/v1/best_model.pth`

### "Image count doesn't match expected"

**Issue:** Extraction incomplete

**Solution:**
1. Re-extract the zip file
2. Check for errors during extraction
3. Verify with: `find data/processed -name "*.jpg" | wc -l` (should be 2,500)

### "Can't download from Google Drive"

**Issue:** Link requires permissions or quota exceeded

**Alternative solutions:**
1. Request access from project maintainer
2. Use Kaggle dataset + preparation notebook
3. Use Hugging Face Hub version (if available)

---

## 📞 Support

If you encounter issues downloading assets:

1. Check this troubleshooting section
2. Verify your internet connection and disk space
3. Try alternative download methods (Kaggle, Hugging Face)
4. Contact project maintainer with:
   - Error message
   - Operating system
   - Steps you've tried

---

## 🔒 File Integrity

**Optional:** Verify file integrity using checksums

```bash
# If checksums are provided
sha256sum -c checksums.txt

# Expected output:
# dataset.zip: OK
# model_v1.zip: OK
```

**Checksums (replace with actual values):**
```
# SHA256 checksums
REPLACE_WITH_ACTUAL_HASH  dataset.zip
REPLACE_WITH_ACTUAL_HASH  model_v1.zip
```

---

## ✅ Next Steps

Once all assets are downloaded and verified:

1. ✅ Dataset extracted to `data/processed/`
2. ✅ Model files in `models/v1/`
3. ✅ Verification script passed

**Proceed to:** [Quick Start Guide in README.md](README.md#quick-start)

**Register the model:**
```bash
# Start MLflow first
mlflow-start  # or manual command from README

# Register model (in another terminal)
python scripts/register_model.py
python scripts/set_production.py
```

Then follow the remaining Quick Start steps to launch the system! 🚀