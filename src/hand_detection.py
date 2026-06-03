import cv2
import mediapipe as mp
import numpy as np

def detect_hands():

    # ── MediaPipe Setup ──────────────────────
    # mp_hands gives us the hand detection tools
    mp_hands = mp.solutions.hands

    # mp_drawing helps us draw the landmarks
    mp_drawing = mp.solutions.drawing_utils

    # mp_drawing_styles gives nice colored lines
    mp_drawing_styles = mp.solutions.drawing_styles

    # Create the Hand detector object
    # static_image_mode=False → for video (not single images)
    # max_num_hands=1        → detect only 1 hand (simpler)
    # min_detection_confidence=0.7 → 70% sure = hand detected
    # min_tracking_confidence=0.5  → 50% sure = keep tracking
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # ── Webcam Setup ─────────────────────────
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return

    print("✅ Hand Detection Started!")
    print("👉 Show your hand to the camera.")
    print("👉 Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Get frame dimensions (height, width)
        h, w, _ = frame.shape

        # ── MediaPipe needs RGB, OpenCV gives BGR ──
        # Convert BGR → RGB before processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame → detect hands
        results = hands.process(rgb_frame)

        # ── If hand is detected ───────────────────
        if results.multi_hand_landmarks:

            # Loop through each detected hand
            # (we set max=1, so only 1 hand here)
            for hand_landmarks in results.multi_hand_landmarks:

                # ── Draw landmarks on frame ───────
                # This draws the 21 dots and connecting lines
                mp_drawing.draw_landmarks(
                    frame,                              # image to draw on
                    hand_landmarks,                     # landmark data
                    mp_hands.HAND_CONNECTIONS,          # which points to connect
                    mp_drawing_styles.get_default_hand_landmarks_style(),   # dot style
                    mp_drawing_styles.get_default_hand_connections_style()  # line style
                )

                # ── Extract landmark coordinates ──
                # This is what we'll use for training later!
                landmark_list = []

                for id, lm in enumerate(hand_landmarks.landmark):
                    # lm.x and lm.y are between 0.0 and 1.0 (normalized)
                    # Multiply by w and h to get pixel positions
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((cx, cy))

                    # Draw landmark ID number on each point
                    cv2.putText(
                        frame,
                        str(id),            # landmark number
                        (cx, cy),           # position
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,                # font size
                        (0, 255, 255),      # yellow color
                        1                   # thickness
                    )

                # ── Draw Bounding Box around hand ─
                # Find min/max x and y from all landmarks
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]

                x_min = int(min(x_coords)) - 20   # 20px padding
                y_min = int(min(y_coords)) - 20
                x_max = int(max(x_coords)) + 20
                y_max = int(max(y_coords)) + 20

                # Make sure box stays within frame
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                # Draw green rectangle around hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              (0, 255, 0), 2)

                # ── Show landmark count on screen ─
                cv2.putText(
                    frame,
                    f"Landmarks: {len(landmark_list)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # ── Print first 5 landmark coords ─
                # (just to show you what the data looks like)
                print(f"Wrist position: {landmark_list[0]}", end="\r")

        else:
            # No hand detected - show message
            cv2.putText(
                frame,
                "No Hand Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),   # red color
                2
            )

        # Show the frame
        cv2.imshow("Stage 2 - Hand Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Quit pressed.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

# Run
if __name__ == "__main__":
    detect_hands()