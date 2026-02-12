#!/bin/bash
# Setup script for downloading and organizing project assets
# Run this after cloning the repository

set -e  # Exit on error

echo "================================================"
echo "  Refund Classifier - Asset Setup Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create required directories
echo "Creating directory structure..."
mkdir -p data/processed/{train,val,test}
mkdir -p data/inference/{input,output,checkpoints}
mkdir -p models/v1
mkdir -p mlflow_data/{artifacts,mlruns}
mkdir -p logs
echo -e "${GREEN}✓${NC} Directories created"
echo ""

# Check for dataset
echo "Checking for dataset..."
if [ -d "data/processed/train" ] && [ "$(ls -A data/processed/train)" ]; then
    echo -e "${GREEN}✓${NC} Dataset found in data/processed/"
else
    echo -e "${YELLOW}⚠${NC}  Dataset not found"
    echo ""
    echo "Please download the dataset manually:"
    echo "  1. Go to: "https://drive.google.com/drive/folders/1g1V4I3WL8FfXLZfkXqrTYcCR8etojiBY?usp=drive_link"
    echo "  2. Download dataset.zip"
    echo "  3. Extract to: data/processed/"
    echo ""
    echo "Expected structure:"
    echo "  data/processed/"
    echo "    ├── train/    (1,750 images)"
    echo "    ├── val/      (375 images)"
    echo "    ├── test/     (375 images)"
    echo "    └── dataset_info.json"
    echo ""
    read -p "Press Enter after downloading the dataset..."
fi

# Check for trained model
echo ""
echo "Checking for trained model..."
if [ -f "models/v1/best_model.pth" ] && [ -f "models/v1/training_metadata.json" ]; then
    echo -e "${GREEN}✓${NC} Model found in models/v1/"
else
    echo -e "${YELLOW}⚠${NC}  Trained model not found"
    echo ""
    echo "Please download the trained model manually:"
    echo "  1. Go to: "https://drive.google.com/drive/folders/1IQ4wyuTYO0TuQvKg0bIZ3n1kTZQpuvtp?usp=drive_link"
    echo "  2. Download model_v1.zip"
    echo "  3. Extract to: models/v1/"
    echo ""
    echo "Expected files:"
    echo "  models/v1/"
    echo "    ├── best_model.pth           (~50MB)"
    echo "    └── training_metadata.json   (~5KB)"
    echo ""
    read -p "Press Enter after downloading the model..."
fi

# Verify setup
echo ""
echo "Verifying setup..."
echo ""

# Check dataset
TRAIN_COUNT=$(find data/processed/train -name "*.jpg" 2>/dev/null | wc -l)
VAL_COUNT=$(find data/processed/val -name "*.jpg" 2>/dev/null | wc -l)
TEST_COUNT=$(find data/processed/test -name "*.jpg" 2>/dev/null | wc -l)

echo "Dataset images:"
echo "  Train: $TRAIN_COUNT (expected: 1,750)"
echo "  Val:   $VAL_COUNT (expected: 375)"
echo "  Test:  $TEST_COUNT (expected: 375)"

if [ "$TRAIN_COUNT" -ge 1700 ] && [ "$VAL_COUNT" -ge 350 ] && [ "$TEST_COUNT" -ge 350 ]; then
    echo -e "  ${GREEN}✓${NC} Dataset looks good"
else
    echo -e "  ${RED}✗${NC} Dataset incomplete (check counts above)"
fi

# Check model
echo ""
echo "Model files:"
if [ -f "models/v1/best_model.pth" ]; then
    MODEL_SIZE=$(du -h models/v1/best_model.pth | cut -f1)
    echo "  best_model.pth: $MODEL_SIZE"
    echo -e "  ${GREEN}✓${NC} Model checkpoint found"
else
    echo -e "  ${RED}✗${NC} best_model.pth not found"
fi

if [ -f "models/v1/training_metadata.json" ]; then
    echo -e "  ${GREEN}✓${NC} Training metadata found"
else
    echo -e "  ${RED}✗${NC} training_metadata.json not found"
fi

# Summary
echo ""
echo "================================================"
echo "  Setup Summary"
echo "================================================"
echo ""

if [ "$TRAIN_COUNT" -ge 1700 ] && [ -f "models/v1/best_model.pth" ]; then
    echo -e "${GREEN}✓ All required assets are in place${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Install dependencies:"
    echo "     uv sync"
    echo ""
    echo "  2. Start MLflow server:"
    echo "     ABS_PATH=\$(pwd)"
    echo "     mlflow server \\"
    echo "       --backend-store-uri sqlite:///\$ABS_PATH/mlflow_data/mlflow.db \\"
    echo "       --default-artifact-root file://\$ABS_PATH/mlflow_data/artifacts \\"
    echo "       --host 0.0.0.0 --port 5000"
    echo ""
    echo "  3. Register model (in another terminal):"
    echo "     python scripts/register_model.py"
    echo "     python scripts/set_production.py"
    echo ""
    echo "  4. Follow the Quick Start in README.md"
    echo ""
else
    echo -e "${RED}✗ Some assets are missing${NC}"
    echo ""
    echo "Please download missing files from the links above."
    echo "Then re-run this script to verify."
    echo ""
fi

echo "================================================"