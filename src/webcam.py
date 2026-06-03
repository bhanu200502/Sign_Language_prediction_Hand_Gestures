import cv2  

def open_webcam():
    cap = cv2.VideoCapture(0)

    # Check if webcam opened successfully
    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam!")
        print("Fix: Make sure no other app is using the camera.")
        return

    print("✅ Webcam opened successfully!")
    print("👉 Press 'Q' on keyboard to quit.")

    # Keep looping to show live video
    while True:
        ret, frame = cap.read()

        # If frame not read properly, skip it
        if not ret:
            print("❌ Failed to read frame. Exiting...")
            break

        # Flip frame horizontally (mirror effect - feels natural)
        frame = cv2.flip(frame, 1)

        # Resize frame to 70% of original size
        frame_resized = cv2.resize(frame, (frame.shape[1] * 7 // 10, frame.shape[0] * 7 // 10))

        # Show the frame in a window named "Stage 1 - Webcam"
        cv2.imshow("Stage 1 - Webcam", frame_resized)

        # Wait 1ms for key press
        # ord('q') = ASCII value of 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Quit pressed. Closing webcam.")
            break

    # Always release camera and close windows when done
    cap.release()
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    open_webcam()