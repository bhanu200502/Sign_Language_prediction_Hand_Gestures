# ============================================
# STAGE 6: Real-Time Prediction
# Goal: Load trained model, predict ASL
#       gesture from live webcam feed
# ============================================

import cv2
import numpy as np
import os
import time

# ── Suppress warnings ─────────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

# ── MediaPipe Import ──────────────────────────────────
import mediapipe as mp
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── TensorFlow Import ─────────────────────────────────
import tensorflow as tf

# ── SETTINGS ──────────────────────────────────────────
MODEL_PATH  = "models/sign_model.h5"
LABELS_PATH = "data/labels.npy"

# Confidence threshold
# Only show prediction if model is this % sure
CONFIDENCE_THRESHOLD = 0.70   # 70%

# How many frames to average prediction over
# (reduces flickering)
SMOOTH_FRAMES = 5
# ──────────────────────────────────────────────────────


def load_model_and_labels():
    """
    Load trained model and gesture labels
    """
    print("\n" + "="*55)
    print("📂 LOADING MODEL AND LABELS")
    print("="*55)

    # ── Check model file ──────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print(f"   Run Stage 5 first:")
        print(f"   python src/model_training.py")
        return None, None

    # ── Check labels file ─────────────────────
    if not os.path.exists(LABELS_PATH):
        print(f"❌ Labels not found: {LABELS_PATH}")
        print(f"   Run Stage 4 first:")
        print(f"   python src/data_preprocessing.py")
        return None, None

    # ── Load model ────────────────────────────
    print("  ⏳ Loading model...")
    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH)

    print(f"  ✅ Model loaded   : {MODEL_PATH}")
    print(f"  ✅ Labels loaded  : {list(labels)}")
    print(f"  ✅ Total classes  : {len(labels)}")
    print(f"  ✅ Input shape    : {model.input_shape}")
    print(f"  ✅ Output shape   : {model.output_shape}")

    return model, labels


def extract_landmarks(image, hands):
    """
    Extract and normalize 63 landmark values
    from one frame — same as Stage 4
    """
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_lms  = results.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand_lms.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks, dtype=np.float32)

        # ── Normalize (same as Stage 4) ───────
        lm      = landmarks.reshape(21, 3)
        wrist   = lm[0].copy()
        lm      = lm - wrist
        max_val = np.max(np.abs(lm))
        if max_val > 0:
            lm = lm / max_val

        return lm.flatten(), results
    
    return None, results


def draw_prediction_box(frame, letter, confidence,
                         x_min, y_min, x_max, y_max):
    """
    Draw a nice prediction box on the frame
    showing the predicted letter and confidence
    """
    h, w = frame.shape[:2]

    # ── Draw bounding box around hand ─────────
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD \
            else (0, 165, 255)
    cv2.rectangle(frame,
                  (x_min, y_min),
                  (x_max, y_max),
                  color, 3)

    # ── Big letter display box ────────────────
    box_x1 = x_min
    box_y1 = max(0, y_min - 100)
    box_x2 = x_min + 120
    box_y2 = y_min

    # Background rectangle for letter
    cv2.rectangle(frame,
                  (box_x1, box_y1),
                  (box_x2, box_y2),
                  (0, 0, 0), -1)   # filled black

    # Draw predicted letter (BIG)
    cv2.putText(frame,
                letter,
                (box_x1 + 15, box_y2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,              # huge font
                (0, 255, 0),      # green
                5)                # thick

    # Draw confidence bar background
    bar_x    = 10
    bar_y    = h - 60
    bar_w    = 300
    bar_h    = 25

    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)

    # Draw confidence bar fill
    fill_w = int(bar_w * confidence)
    bar_color = (0, 255, 0)   if confidence > 0.8 \
           else (0, 165, 255) if confidence > 0.5 \
           else (0, 0, 255)

    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_h),
                  bar_color, -1)

    # Confidence text
    cv2.putText(frame,
                f"Confidence: {confidence*100:.1f}%",
                (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    return frame


def draw_info_panel(frame, fps, prediction_history):
    """
    Draw info panel at top of screen
    showing FPS and recent predictions
    """
    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 45),
                  (30, 30, 30), -1)

    # FPS
    cv2.putText(frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # Recent predictions history
    history_text = "History: " + " → ".join(
        prediction_history[-5:]
    )
    cv2.putText(frame,
                history_text,
                (120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    # Instructions at bottom
    cv2.putText(frame,
                "ESC: Quit  |  C: Clear History  |  S: Screenshot",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    return frame


def predict_realtime():
    """
    Main real-time prediction loop
    """

    # ── Load model and labels ─────────────────
    model, labels = load_model_and_labels()
    if model is None:
        return

    # ── MediaPipe setup ───────────────────────
    hands = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = 0.7,
        min_tracking_confidence  = 0.5
    )

    # ── Webcam setup ──────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "="*55)
    print("🎯 REAL-TIME PREDICTION STARTED!")
    print("="*55)
    print("  👋 Show your hand to the camera")
    print("  ESC → Quit")
    print("  C   → Clear prediction history")
    print("  S   → Save screenshot")
    print("="*55)

    # ── State variables ───────────────────────
    prediction_history = []    # last N predictions
    recent_preds       = []    # for smoothing
    screenshot_count   = 0

    # FPS calculation
    fps        = 0
    fps_time   = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame!")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── FPS calculation ───────────────────
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps        = frame_count
            frame_count = 0
            fps_time   = time.time()

        # ── Extract landmarks ─────────────────
        landmarks, results = extract_landmarks(frame, hands)

        current_letter     = "?"
        current_confidence = 0.0

        if landmarks is not None and \
           results.multi_hand_landmarks:

            # ── Get bounding box ──────────────
            hand_lms  = results.multi_hand_landmarks[0]
            x_coords  = [lm.x * w for lm in hand_lms.landmark]
            y_coords  = [lm.y * h for lm in hand_lms.landmark]
            x_min = max(0,   int(min(x_coords)) - 30)
            y_min = max(0,   int(min(y_coords)) - 30)
            x_max = min(w,   int(max(x_coords)) + 30)
            y_max = min(h,   int(max(y_coords)) + 30)

            # ── Draw landmarks ────────────────
            mp_drawing.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # ── Predict ───────────────────────
            input_data = landmarks.reshape(1, -1)
            prediction = model.predict(
                input_data, verbose=0
            )[0]

            pred_idx    = np.argmax(prediction)
            confidence  = float(prediction[pred_idx])
            pred_letter = str(labels[pred_idx])

            # ── Smooth predictions ────────────
            # Average over last SMOOTH_FRAMES frames
            recent_preds.append(pred_letter)
            if len(recent_preds) > SMOOTH_FRAMES:
                recent_preds.pop(0)

            # Most common prediction in recent frames
            if recent_preds:
                current_letter     = max(
                    set(recent_preds),
                    key=recent_preds.count
                )
                current_confidence = confidence

            # ── Add to history ────────────────
            if confidence >= CONFIDENCE_THRESHOLD:
                if (not prediction_history or
                    prediction_history[-1] != current_letter):
                    prediction_history.append(current_letter)
                    if len(prediction_history) > 20:
                        prediction_history.pop(0)

            # ── Draw prediction box ───────────
            frame = draw_prediction_box(
                frame,
                current_letter,
                current_confidence,
                x_min, y_min,
                x_max, y_max
            )

        else:
            # No hand detected
            cv2.putText(frame,
                "Show your hand...",
                (w//2 - 150, h//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2)

        # ── Draw info panel ───────────────────
        frame = draw_info_panel(frame, fps,
                                prediction_history)

        # ── Show frame ────────────────────────
        cv2.imshow("Sign Language Prediction", frame)

        # ── Key handling ──────────────────────
        key = cv2.waitKey(1) & 0xFF

        # ESC = quit
        if key == 27:
            print("\n👋 Quit pressed.")
            break

        # C = clear history
        elif key == ord('c'):
            prediction_history.clear()
            recent_preds.clear()
            print("🗑️  History cleared!")

        # S = screenshot
        elif key == ord('s'):
            screenshot_count += 1
            fname = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸 Screenshot saved: {fname}")

    # ── Cleanup ───────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # ── Final word history ────────────────────
    if prediction_history:
        print("\n" + "="*55)
        print("📝 YOUR PREDICTION HISTORY:")
        print("="*55)
        print("  Letters : " + " → ".join(prediction_history))
        word = "".join(prediction_history)
        print(f"  Word    : {word}")
        print("="*55)


# ── RUN ───────────────────────────────────────────────
if __name__ == "__main__":

    print("="*55)
    print("  STAGE 6 — REAL-TIME PREDICTION")
    print("="*55)

    # Check running from root
    if not os.path.exists("models"):
        print("❌ Run from project root:")
        print("   cd D:\\SIGN_LANGUAGE")
        print("   python src/prediction.py")
    else:
        predict_realtime()