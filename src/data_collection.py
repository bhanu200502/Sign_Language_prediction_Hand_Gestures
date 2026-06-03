# ============================================
# STAGE 3 REPLACEMENT: Kaggle Dataset Setup
# Goal: Copy Kaggle ASL dataset into our
#       project data/ folder properly
# ============================================

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# ── SETTINGS ──────────────────────────────────────────

# ✏️ CHANGE THIS to where you extracted the ZIP
# Examples:
# Windows: "C:/Users/Tony/Downloads/asl_alphabet_train/asl_alphabet_train"
# VS Code: just drag the folder and copy path
KAGGLE_PATH =r"D:\SIGN_LANGUAGE\data\asl_alphabet_train\asl_alphabet_train"

# Our project data folder
DATA_PATH = "data"

# How many images to copy per gesture
# Kaggle has 3000 per class — we use 300 to keep it fast
# More = better accuracy but slower training
IMAGES_PER_CLASS = 300

# Only use A-Z (skip 'space', 'del', 'nothing' for now)
# You can add them back later if you want!
GESTURES = ['A', 'B', 'C', 'D', 'E', 'F',
            'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z']

# Image size for saving
IMG_SIZE = 224
# ──────────────────────────────────────────────────────


def verify_kaggle_path():
    """Check if Kaggle dataset path is correct"""

    print("\n" + "="*55)
    print("🔍 VERIFYING KAGGLE DATASET PATH")
    print("="*55)

    # Check if path exists
    if not os.path.exists(KAGGLE_PATH):
        print(f"❌ ERROR: Path not found!")
        print(f"   Path: {KAGGLE_PATH}")
        print(f"\n💡 FIX:")
        print(f"   1. Extract your ZIP file first")
        print(f"   2. Find the folder with A/, B/, C/ inside")
        print(f"   3. Copy that full path into KAGGLE_PATH above")
        return False

    # Check if gesture folders exist inside
    found = []
    missing = []
    for g in GESTURES:
        folder = os.path.join(KAGGLE_PATH, g)
        if os.path.exists(folder):
            count = len(os.listdir(folder))
            found.append(f"{g}({count})")
        else:
            missing.append(g)

    print(f"✅ Path found: {KAGGLE_PATH}")
    print(f"✅ Gesture folders found: {found[:5]}...")

    if missing:
        print(f"⚠️  Missing folders: {missing}")

    if len(found) == 0:
        print("❌ No gesture folders found in this path!")
        print("   Make sure you're pointing to the RIGHT folder")
        print("   The folder should contain A/, B/, C/ directly")
        return False

    print(f"\n✅ Dataset verified! Found {len(found)} gesture folders")
    return True


def copy_and_resize_images():
    """
    Copy images from Kaggle dataset into our data/ folder
    Resize to IMG_SIZE × IMG_SIZE while copying
    """

    print("\n" + "="*55)
    print("📋 COPYING KAGGLE IMAGES TO PROJECT")
    print("="*55)
    print(f"📌 Source : {KAGGLE_PATH}")
    print(f"📌 Target : {DATA_PATH}/")
    print(f"📌 Per class: {IMAGES_PER_CLASS} images")
    print(f"📌 Image size: {IMG_SIZE}×{IMG_SIZE}")
    print("="*55)

    total_copied  = 0
    total_skipped = 0

    for gesture in GESTURES:

        # Source folder (Kaggle)
        src_folder = os.path.join(KAGGLE_PATH, gesture)

        # Target folder (our project)
        dst_folder = os.path.join(DATA_PATH, gesture)

        # Skip if Kaggle folder doesn't exist
        if not os.path.exists(src_folder):
            print(f"⚠️  Skipping '{gesture}' — not in Kaggle dataset")
            continue

        # Create our target folder
        os.makedirs(dst_folder, exist_ok=True)

        # Get all images in Kaggle folder
        all_images = [
            f for f in os.listdir(src_folder)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Randomly pick IMAGES_PER_CLASS images
        # (so we get variety, not just first 300)
        random.shuffle(all_images)
        selected = all_images[:IMAGES_PER_CLASS]

        print(f"\n📂 '{gesture}': copying {len(selected)} images...")

        copied  = 0
        skipped = 0

        for i, img_file in enumerate(selected):

            src_path = os.path.join(src_folder, img_file)
            dst_name = f"{gesture}_{i+1:04d}.jpg"
            dst_path = os.path.join(dst_folder, dst_name)

            # Load image
            img = cv2.imread(src_path)

            if img is None:
                skipped += 1
                continue

            # Resize to our standard size
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Save to our data folder
            cv2.imwrite(dst_path, img_resized)
            copied += 1

            # Show progress every 50 images
            if (i + 1) % 50 == 0:
                print(f"   → {i+1}/{len(selected)} done...")

        print(f"   ✅ Copied: {copied}  ⚠️  Skipped: {skipped}")
        total_copied  += copied
        total_skipped += skipped

    # ── Final Summary ──────────────────────────
    print("\n" + "="*55)
    print("📊 COPY COMPLETE!")
    print("="*55)
    print(f"✅ Total images copied  : {total_copied}")
    print(f"⚠️  Total images skipped : {total_skipped}")
    print(f"\n📁 Your data/ folder now has:")

    for gesture in GESTURES:
        folder = os.path.join(DATA_PATH, gesture)
        if os.path.exists(folder):
            n = len(os.listdir(folder))
            bar = "█" * (n // 10)
            print(f"   {gesture}/ → {bar} {n} images")

    print("\n✅ Dataset ready! Now run Stage 4 →")
    print("   python src/stage4_preprocessing.py")


def show_sample_images():
    """
    Show a few sample images from dataset
    so you can verify they look correct
    """
    print("\n👀 Showing sample images (press any key to continue)...")

    for gesture in GESTURES[:5]:   # show first 5 gestures
        folder = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(folder):
            continue

        images = os.listdir(folder)
        if not images:
            continue

        # Load first image
        img_path = os.path.join(folder, images[0])
        img = cv2.imread(img_path)

        if img is not None:
            # Add label text
            cv2.putText(img, f"Gesture: {gesture}",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)

            cv2.imshow(f"Sample - {gesture}", img)
            cv2.waitKey(1000)   # show for 1 second
            cv2.destroyAllWindows()


# ── MAIN ──────────────────────────────────────────────
if __name__ == "__main__":

    print("\n🚀 KAGGLE ASL DATASET SETUP")
    print("This replaces Stage 3 (Data Collection)")

    # Step 1: Verify path is correct
    if not verify_kaggle_path():
        print("\n❌ Fix the KAGGLE_PATH first, then run again!")
        exit()

    # Step 2: Copy and resize images
    copy_and_resize_images()

    # Step 3: Show samples to verify
    show_sample_images()

    print("\n🎉 ALL DONE! Your dataset is ready!")
    print("👉 Next: python src/stage4_preprocessing.py")