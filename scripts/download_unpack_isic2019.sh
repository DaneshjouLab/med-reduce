#!/bin/bash
set -e

# -------------------- Config --------------------
BASE_DIR="/oak/stanford/groups/roxanad/isic2019"
TRAIN_DIR="$BASE_DIR/data/train"
TEST_DIR="$BASE_DIR/data/test"
CHECKSUM_FILE="$BASE_DIR/checksums.sha256"

mkdir -p "$TRAIN_DIR/images" "$TEST_DIR/images"
cd "$BASE_DIR"

# -------------------- URLs --------------------
TRAIN_IMAGES_URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
TRAIN_LABELS_URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
TEST_IMAGES_URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip"
TEST_LABELS_URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_GroundTruth.csv"

# -------------------- Download with Progress + Fail --------------------
download() {
    local url=$1
    local out=$2
    echo "Downloading: $out"
    curl -fL# "$url" -o "$out" || {
        echo "❌ Failed to download $out"
        exit 1
    }
}

download "$TRAIN_IMAGES_URL" ISIC_2019_Training_Input.zip
download "$TRAIN_LABELS_URL" ISIC_2019_Training_GroundTruth.csv
download "$TEST_IMAGES_URL" ISIC_2019_Test_Input.zip
download "$TEST_LABELS_URL" ISIC_2019_Test_GroundTruth.csv

# -------------------- Checksum (Optional) --------------------
echo "Generating SHA256 checksums..."
sha256sum ISIC_2019_Training_Input.zip \
          ISIC_2019_Training_GroundTruth.csv \
          ISIC_2019_Test_Input.zip \
          ISIC_2019_Test_GroundTruth.csv > "$CHECKSUM_FILE"

echo "Verifying checksums..."
sha256sum -c "$CHECKSUM_FILE" || {
    echo "❌ Checksum verification failed."
    exit 1
}

# -------------------- Unpack and Organize --------------------
echo "Unpacking training images..."
unzip -q ISIC_2019_Training_Input.zip -d "$TRAIN_DIR/images"
echo "Moving training labels..."
mv ISIC_2019_Training_GroundTruth.csv "$TRAIN_DIR/labels.csv"

echo "Unpacking test images..."
unzip -q ISIC_2019_Test_Input.zip -d "$TEST_DIR/images"
echo "Moving test labels..."
mv ISIC_2019_Test_GroundTruth.csv "$TEST_DIR/labels.csv"

# -------------------- Cleanup --------------------
rm ISIC_2019_Training_Input.zip ISIC_2019_Test_Input.zip

echo "✅ Dataset ready at: $BASE_DIR"
