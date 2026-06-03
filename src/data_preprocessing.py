# ============================================
# STAGE 4: Data Preprocessing
# REWRITTEN — Works with all MediaPipe versions
# ============================================

import cv2
import numpy as np
import os
import time

# ── MediaPipe Import (version-safe) ───────────────────
try:
    import mediapipe as mp
    mp_hands    = mp.solutions.hands
    mp_drawing  = mp.solutions.drawing_utils
    MEDIAPIPE_OK = True
    print(f"✅ MediaPipe version: {mp.__version__}")
except AttributeError:
    print("⚠️  MediaPipe solutions not found!")
    print("    Run: pip install mediapipe==0.10.3")
    MEDIAPIPE_OK = False
except ImportError:
    print("❌ MediaPipe not installed!")
    print("    Run: pip install mediapipe==0.10.3")
    MEDIAPIPE_OK = False

# ── SETTINGS ──────────────────────────────────────────
DATA_PATH = "data"      # where A/, B/, C/ folders are
SAVE_PATH = "data"      # where to save .npy files
IMG_SIZE  = 224         # image size used in Stage 3

GESTURES = ['A', 'B', 'C', 'D', 'E', 'F',
            'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z']
# ──────────────────────────────────────────────────────


def get_hands_detector():
    """
    Create MediaPipe hands detector safely.
    Works with both old and new MediaPipe versions.
    """
    try:
        hands = mp_hands.Hands(
            static_image_mode        = True,
            max_num_hands            = 1,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5
        )
        return hands
    except Exception as e:
        print(f"❌ Error creating hands detector: {e}")
        print("   Fix: pip uninstall mediapipe -y")
        print("        pip install mediapipe==0.10.3")
        return None


def extract_landmarks(image, hands):
    """
    Extract 63 landmark values from one image.

    Input  : BGR image (numpy array)
    Output : numpy array of 63 floats  OR  None
    """
    try:
        # MediaPipe needs RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hand
        results = hands.process(rgb)

        # If hand found
        if results.multi_hand_landmarks:
            hand_lms  = results.multi_hand_landmarks[0]
            landmarks = []

            for lm in hand_lms.landmark:
                # Each landmark = x, y, z (normalized 0-1)
                landmarks.extend([lm.x, lm.y, lm.z])

            # 21 landmarks × 3 = 63 values
            return np.array(landmarks, dtype=np.float32)

    except Exception as e:
        pass  # silently skip bad images

    return None


def normalize_landmarks(landmarks):
    """
    Normalize landmarks so hand position
    on screen doesn't affect prediction.

    - Make wrist = center (0, 0, 0)
    - Scale all values to [-1, +1]
    """
    try:
        # Reshape to (21, 3)
        lm = landmarks.reshape(21, 3)

        # Move wrist to origin
        wrist = lm[0].copy()
        lm    = lm - wrist

        # Scale to [-1, +1]
        max_val = np.max(np.abs(lm))
        if max_val > 0:
            lm = lm / max_val

        return lm.flatten().astype(np.float32)

    except Exception as e:
        return landmarks  # return as-is if normalization fails


def check_data_folder():
    """
    Check which gesture folders exist
    and how many images each has.
    """
    print("\n" + "="*55)
    print("📁 CHECKING DATA FOLDER")
    print("="*55)

    total          = 0
    found_gestures = []

    for gesture in GESTURES:
        folder = os.path.join(DATA_PATH, gesture)

        if os.path.exists(folder):
            images = [
                f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg','.jpeg','.png'))
            ]
            n = len(images)

            if n > 0:
                bar = "█" * (n // 10)
                print(f"  ✅ {gesture}/ → {bar} ({n} images)")
                total += n
                found_gestures.append(gesture)
            else:
                print(f"  ⚠️  {gesture}/ → empty folder")
        else:
            print(f"  ❌ {gesture}/ → not found")

    print(f"\n  📊 Total images   : {total}")
    print(f"  📊 Total gestures : {len(found_gestures)}")

    if total == 0:
        print("\n  ❌ No images found in data/ folder!")
        print("  👉 Run Stage 3 first:")
        print("     python src/data_collection.py")
        return False, []

    return True, found_gestures


def preprocess_dataset():
    """
    Full preprocessing pipeline:
    1. Check data folder
    2. Extract landmarks from each image
    3. Normalize landmarks
    4. Save X.npy, y.npy, labels.npy
    """

    # ── Check MediaPipe ───────────────────────
    if not MEDIAPIPE_OK:
        print("❌ MediaPipe not working!")
        print("   Run: pip install mediapipe==0.10.3")
        return

    # ── Check data folder ─────────────────────
    ok, found_gestures = check_data_folder()
    if not ok:
        return

    # ── Create hands detector ─────────────────
    hands = get_hands_detector()
    if hands is None:
        return

    # ── Storage ───────────────────────────────
    X               = []   # features
    y               = []   # labels
    total_processed = 0
    total_skipped   = 0
    gesture_counts  = {}
    collected_labels = []

    print("\n" + "="*55)
    print("🔧 EXTRACTING LANDMARKS")
    print("="*55)

    start_time = time.time()

    # ── Loop each gesture ─────────────────────
    for label_idx, gesture in enumerate(found_gestures):

        folder = os.path.join(DATA_PATH, gesture)

        image_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ]

        print(f"\n📂 '{gesture}' — {len(image_files)} images")

        success = 0
        skipped = 0

        # ── Loop each image ───────────────────
        for i, img_file in enumerate(image_files):

            img_path = os.path.join(folder, img_file)

            # Load image
            image = cv2.imread(img_path)

            # Skip unreadable images
            if image is None:
                skipped += 1
                continue

            # Resize to standard size
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            # Extract 63 landmarks
            landmarks = extract_landmarks(image, hands)

            if landmarks is not None:
                # Normalize
                landmarks_norm = normalize_landmarks(landmarks)

                # Store
                X.append(landmarks_norm)
                y.append(label_idx)

                success += 1
                total_processed += 1
            else:
                skipped += 1
                total_skipped += 1

            # Progress update every 100 images
            if (i + 1) % 100 == 0:
                pct = ((i + 1) / len(image_files)) * 100
                print(f"   → {i+1}/{len(image_files)}"
                      f" ({pct:.0f}%)"
                      f" ✅{success} ⚠️{skipped}")

        gesture_counts[gesture]  = success
        collected_labels.append(gesture)

        print(f"   ✅ Extracted : {success}")
        print(f"   ⚠️  Skipped  : {skipped}")

    # ── Close MediaPipe ───────────────────────
    try:
        hands.close()
    except:
        pass

    # ── Validate results ──────────────────────
    if len(X) == 0:
        print("\n❌ ERROR: Zero landmarks extracted!")
        print("   Possible fixes:")
        print("   1. Lower confidence: change 0.5 → 0.3")
        print("   2. Check images have visible hands")
        print("   3. pip install mediapipe==0.10.3")
        return

    # ── Convert to numpy ──────────────────────
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────
    print("\n" + "="*55)
    print("📊 PREPROCESSING COMPLETE!")
    print("="*55)
    print(f"  ✅ Extracted    : {total_processed} samples")
    print(f"  ⚠️  Skipped     : {total_skipped} images")
    print(f"  📐 X shape      : {X.shape}")
    print(f"     {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  📐 y shape      : {y.shape}")
    print(f"  🏷️  Classes      : {len(np.unique(y))}")
    print(f"  ⏱️  Time         : {elapsed:.1f} sec")

    # ── Save fi        les ────────────────────────────
    os.makedirs(SAVE_PATH, exist_ok=True)

    X_path      = os.path.join(SAVE_PATH, "X.npy")
    y_path      = os.path.join(SAVE_PATH, "y.npy")
    labels_path = os.path.join(SAVE_PATH, "labels.npy")

    np.save(X_path,      X)
    np.save(y_path,      y)
    np.save(labels_path, np.array(collected_labels))

    print(f"\n  💾 X.npy      → {X_path}")
    print(f"  💾 y.npy      → {y_path}")
    print(f"  💾 labels.npy → {labels_path}")

    # ── Label map ─────────────────────────────
    print("\n📋 LABEL MAPPING:")
    print("-"*40)
    for i, g in enumerate(collected_labels):
        n   = gesture_counts.get(g, 0)
        bar = "█" * (n // 10)
        print(f"  {i:2d} → '{g}'  {bar} ({n})")
    print("="*55)

    # ── Verify values ─────────────────────────
    print("\n🔍 VERIFY:")
    print(f"  X[0][:9]  = {X[0][:9].round(4)}")
    print(f"  X min/max = {X.min():.4f} / {X.max():.4f}")
    print(f"  y[:10]    = {y[:10]}")
    print(f"\n🎉 Done! Now run Stage 5:")
    print(f"   python src/model_training.py")


# ── RUN ───────────────────────────────────────────────
if __name__ == "__main__":

    print("="*55)
    print("  STAGE 4 — DATA PREPROCESSING")
    print("="*55)

    # Check running from correct folder
    if not os.path.exists("data"):
        print("❌ 'data/' folder not found!")
        print("   Make sure you run from project root:")
        print("   cd D:\\SIGN_LANGUAGE")
        print("   python src/data_preprocessing.py")
    else:
        preprocess_dataset()

