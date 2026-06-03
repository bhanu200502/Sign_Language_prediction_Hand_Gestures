# ============================================
# SIGN LANGUAGE PREDICTION - STREAMLIT APP
# Run: streamlit run app.py
# Opens: http://localhost:8501
# ============================================

import streamlit as st
import cv2
import numpy as np
import os
import time
import tempfile
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "sign_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "src", "data", "labels.npy")
TRAINING_PLOT_PATH = os.path.join(BASE_DIR, "src", "models", "training_plot.png")
TEST_ACCURACY_PATH = os.path.join(BASE_DIR, "src", "models", "test_accuracy.npy")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "src", "models", "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(BASE_DIR, "src", "models", "roc_curve.png")
PR_CURVE_PATH = os.path.join(BASE_DIR, "src", "models", "precision_recall_curve.png")
METRICS_SUMMARY_PATH = os.path.join(BASE_DIR, "src", "models", "performance_metrics.txt")

# ── Suppress warnings ─────────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title = "Sign Language Prediction",
    page_icon  = "🤟",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top right, rgba(0, 212, 255, 0.08), transparent 28%),
            linear-gradient(180deg, #0b1020 0%, #0e1117 55%, #09111d 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151826 0%, #1b1f2c 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #00d4ff;
        text-shadow: 0 0 20px #00d4ff;
        margin-bottom: 0;
    }

    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-top: 0;
    }

    /* Prediction box */
    .pred-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0,212,255,0.3);
    }

    .pred-letter {
        font-size: 8rem;
        font-weight: bold;
        color: #00ff88;
        line-height: 1;
    }

    .pred-conf {
        font-size: 1.2rem;
        color: #aaa;
    }

    /* Status cards */
    .status-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #00d4ff;
        margin: 5px 0;
    }

    /* Info card */
    .info-card {
        background: #16213e;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(20, 31, 61, 0.92));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        color: #7dd3fc;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
    }

    .hero-copy {
        color: #cbd5e1;
        font-size: 1rem;
        margin-top: 0.75rem;
        margin-bottom: 0;
    }

    .setup-card {
        background: linear-gradient(180deg, rgba(69, 26, 26, 0.9), rgba(55, 22, 22, 0.88));
        border: 1px solid rgba(248, 113, 113, 0.28);
        border-radius: 18px;
        padding: 1.25rem;
        margin-top: 1rem;
    }

    .setup-card h3 {
        color: #fecaca;
        margin-top: 0;
        margin-bottom: 0.75rem;
    }

    .setup-card p {
        color: #fee2e2;
        margin-bottom: 0.55rem;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1rem;
        min-height: 124px;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .metric-copy {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════

@st.cache_resource
def load_mediapipe():
    """Load MediaPipe (cached — loads only once)"""
    import mediapipe as mp
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    hands      = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = 0.7,
        min_tracking_confidence  = 0.5
    )
    return hands, mp_hands, mp_drawing, mp_styles


@st.cache_resource
def load_model():
    """Load trained model (cached — loads only once)"""
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        return None, None

    if not os.path.exists(LABELS_PATH):
        return None, None

    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH)
    return model, labels


def extract_and_normalize(frame, hands, mp_hands,
                           mp_drawing, mp_styles):
    """
    Extract 63 landmarks from frame,
    draw landmarks, return landmarks + annotated frame
    """
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks  = None
    x_min = y_min = x_max = y_max = 0

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_lms,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        # Extract landmarks
        lm_list = []
        for lm in hand_lms.landmark:
            lm_list.extend([lm.x, lm.y, lm.z])

        lm_arr = np.array(lm_list, dtype=np.float32)

        # Normalize
        lm_r  = lm_arr.reshape(21, 3)
        wrist = lm_r[0].copy()
        lm_r  = lm_r - wrist
        mx    = np.max(np.abs(lm_r))
        if mx > 0:
            lm_r = lm_r / mx

        landmarks = lm_r.flatten()

        # Bounding box
        x_coords = [lm.x * w for lm in hand_lms.landmark]
        y_coords = [lm.y * h for lm in hand_lms.landmark]
        x_min = max(0,   int(min(x_coords)) - 30)
        y_min = max(0,   int(min(y_coords)) - 30)
        x_max = min(w,   int(max(x_coords)) + 30)
        y_max = min(h,   int(max(y_coords)) + 30)

        # Draw bounding box
        cv2.rectangle(frame,
                      (x_min, y_min),
                      (x_max, y_max),
                      (0, 255, 0), 2)

    return landmarks, frame, (x_min, y_min, x_max, y_max)


def check_project_status():
    """Check all files and return status dict"""
    gestures = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    gesture_count = 0
    total_images  = 0

    for g in gestures:
        folder = os.path.join(BASE_DIR, "src", "data", g)
        if os.path.exists(folder):
            n = len([
                f for f in os.listdir(folder)
                if f.endswith(('.jpg','.jpeg','.png'))
            ])
            if n > 0:
                gesture_count += 1
                total_images  += n

    # Load test accuracy if available
    test_accuracy = None
    if os.path.exists(TEST_ACCURACY_PATH):
        try:
            test_accuracy = float(np.load(TEST_ACCURACY_PATH))
        except:
            test_accuracy = None

    return {
        "dataset"     : gesture_count > 0,
        "gesture_count": gesture_count,
        "total_images" : total_images,
        "X_npy"       : os.path.exists(os.path.join(BASE_DIR, "src", "data", "X.npy")),
        "y_npy"       : os.path.exists(os.path.join(BASE_DIR, "src", "data", "y.npy")),
        "labels_npy"  : os.path.exists(LABELS_PATH),
        "model"       : os.path.exists(MODEL_PATH),
        "plot"        : os.path.exists(TRAINING_PLOT_PATH),
        "test_accuracy": test_accuracy,
    }


def render_metric_card(label, value, copy):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <p class="metric-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_setup_state():
    status = check_project_status()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Live prediction</div>
            <h1 class="hero-title">Train the model once, then come straight back here.</h1>
            <p class="hero-copy">
                The page now keeps a clean layout and shows what is missing instead of leaving a giant empty space.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="setup-card">
                <h3>Model files are missing</h3>
                <p>Run the training script from the project folder.</p>
                <p>After that, refresh this page and the webcam panel will appear automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code("python src/model_training.py", language="bash")
        st.info("Expected files: `models/sign_model.h5` and `data/labels.npy`.")

    with right:
        render_metric_card(
            "Dataset",
            f"{status['gesture_count']}/26",
            f"{status['total_images']} images found in the dataset folders.",
        )
        render_metric_card(
            "Arrays",
            "Ready" if status["X_npy"] and status["y_npy"] and status["labels_npy"] else "Pending",
            "Checks for X.npy, y.npy, and labels.npy.",
        )
        render_metric_card(
            "Model",
            "Found" if status["model"] else "Missing",
            "Live prediction unlocks when the trained model file exists.",
        )


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤟 Sign Language AI")
        st.markdown("---")

        # Student info
        st.markdown("### 👨‍🎓 Student Info")
        st.markdown("""
        - **Name:** P. Bhanu Prakash
        - **ID:** 23C11A6609
        - **Dept:** CSE (AI & ML)
        - **College:** Anurag Engineering
        """)

        st.markdown("---")

        # Navigation
        st.markdown("### 📌 Navigation")
        page = st.radio(
            "Go to:",
            ["🏠 Home",
             "🎯 Live Prediction",
             "📸 Image Prediction",
             "📊 Project Status",
             "📈 Model Metrics",
             "ℹ️  About"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")
        confidence = st.slider(
            "Confidence Threshold",
            min_value = 0.3,
            max_value = 1.0,
            value     = 0.7,
            step      = 0.05,
            help      = "Minimum confidence to show prediction"
        )

        smooth = st.slider(
            "Prediction Smoothing",
            min_value = 1,
            max_value = 15,
            value     = 5,
            help      = "Higher = more stable but slower"
        )

        st.markdown("---")
        st.caption("Sign Language Prediction v1.0")
        st.caption("Powered by MediaPipe + TensorFlow")

    return page, confidence, smooth


# ══════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════

def page_home():
    """Home page"""

    # Title
    st.markdown(
        '<h1 class="main-title">🤟 Sign Language Prediction</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-title">Using Hand Gestures & AI — ASL Alphabet Recognition</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>📷 Real-Time Detection</h3>
        <p>Uses your webcam to detect hand gestures
        in real-time with MediaPipe</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>🧠 AI Powered</h3>
        <p>Deep Neural Network trained on
        87,000 ASL gesture images</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
        <h3>⚡ Fast & Accurate</h3>
        <p>Predicts 26 ASL letters (A-Z)
        with high accuracy in real-time</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # How it works
    st.markdown("## 🔄 How It Works")

    steps = [
        ("📷", "Webcam captures your hand"),
        ("🖐️", "MediaPipe detects 21 landmarks"),
        ("🔢", "63 values (x,y,z × 21) extracted"),
        ("🧠", "Neural Network predicts letter"),
        ("✅", "Predicted ASL letter shown!")
    ]

    cols = st.columns(5)
    for i, (icon, text) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center;
                        background:#1a1a2e;
                        border-radius:10px;
                        padding:15px;
                        border:1px solid #333'>
                <div style='font-size:2rem'>{icon}</div>
                <div style='font-size:0.8rem;
                            color:#aaa;
                            margin-top:8px'>{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick start
    st.markdown("## 🚀 Quick Start")
    st.info("""
    👈 Use the **sidebar** to navigate:
    - **Live Prediction** → Real-time webcam prediction
    - **Image Prediction** → Upload an image to predict
    - **Project Status** → Check if everything is ready
    """)


def page_live_prediction(confidence_threshold, smooth_frames):
    """Live webcam prediction page"""

    st.markdown("## Live Sign Language Prediction")
    st.caption("Use your webcam to detect a hand sign and convert it into an ASL letter.")

    model, labels = load_model()

    if model is None:
        render_model_setup_state()
        return

    # Load and display test accuracy
    status = check_project_status()
    if status['test_accuracy'] is not None:
        st.success(f"Model loaded successfully. Recognizes {len(labels)} gestures. Test Accuracy: {status['test_accuracy']*100:.2f}%")
    else:
        st.success(f"Model loaded successfully. Recognizes {len(labels)} gestures.")

    # Keyboard Shortcuts Information
    with st.expander("⌨️ Keyboard Shortcuts & Options"):
        st.markdown("""
        ### 🎮 Desktop Webcam App Shortcuts
        If you run the desktop version using `python src/prediction.py`, the following keyboard shortcuts are available:
        
        | Key | Action |
        |-----|--------|
        | **ESC** | Quit the webcam feed |
        | **C** | Clear prediction history |
        | **S** | Save a screenshot of the current frame |
        
        ### 🌐 Streamlit Interface Controls
        In this web interface, use these buttons instead:
        - **Clear Word** - Clear all predictions (replaces C key)
        - **Add Space** - Add a space to the word (useful for separating words)
        - Toggle **Start Camera** to start/stop the webcam feed (replaces ESC key)
        
        The **Clear Word** and **Add Space** buttons are located in the right panel under "Word Builder".
        """)

    st.markdown("---")

    col1, col2 = st.columns([1.7, 1], gap="large")

    with col2:
        st.markdown("### Prediction")
        pred_placeholder = st.empty()
        conf_placeholder = st.empty()
        hist_placeholder = st.empty()

        st.markdown("---")
        st.markdown("### Word Builder")
        word_placeholder = st.empty()

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Clear Word", use_container_width=True):
                st.session_state.word = ""
                st.session_state.history = []

        with col_btn2:
            if st.button("Add Space", use_container_width=True):
                if "word" in st.session_state:
                    st.session_state.word += " "

        # Screenshot button
        screenshot_col1, screenshot_col2 = st.columns(2)
        with screenshot_col1:
            st.markdown("**📸 Screenshot**")
        with screenshot_col2:
            screenshot_btn = st.empty()

        st.markdown("---")
        st.markdown("### Stats")
        fps_placeholder = st.empty()
        stat_placeholder = st.empty()

    with col1:
        st.markdown("### Camera Feed")
        run_camera = st.toggle("Start Camera", value=False)
        frame_placeholder = st.empty()

    if "word" not in st.session_state:
        st.session_state.word = ""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest_frame" not in st.session_state:
        st.session_state.latest_frame = None

    if run_camera:
        hands, mp_hands, mp_drawing, mp_styles = load_mediapipe()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Cannot open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        recent_preds = []
        frame_count = 0
        fps = 0
        fps_time = time.time()
        total_preds = 0
        screenshot_counter = 0

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()

            landmarks, frame, bbox = extract_and_normalize(frame, hands, mp_hands, mp_drawing, mp_styles)

            current_letter = "-"
            current_confidence = 0.0

            if landmarks is not None:
                pred = model.predict(landmarks.reshape(1, -1), verbose=0)[0]
                idx = np.argmax(pred)
                conf = float(pred[idx])
                letter = str(labels[idx])

                recent_preds.append(letter)
                if len(recent_preds) > smooth_frames:
                    recent_preds.pop(0)

                current_letter = max(set(recent_preds), key=recent_preds.count)
                current_confidence = conf

                if conf >= confidence_threshold:
                    total_preds += 1
                    if not st.session_state.history or st.session_state.history[-1] != current_letter:
                        st.session_state.history.append(current_letter)
                        st.session_state.word += current_letter

                cv2.putText(
                    frame,
                    f"{current_letter} ({conf * 100:.0f}%)",
                    (bbox[0], max(30, bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )

            cv2.putText(
                frame,
                f"FPS: {fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Store the frame for screenshot download
            st.session_state.latest_frame = frame.copy()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            pred_placeholder.markdown(
                f"""
                <div class="pred-box">
                    <div class="pred-letter" style="color:{'#00ff88' if current_confidence >= confidence_threshold else '#ff8800'}">
                        {current_letter}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            conf_placeholder.progress(
                min(1.0, current_confidence),
                text=f"Confidence: {current_confidence * 100:.1f}%",
            )

            recent_history = " -> ".join(st.session_state.history[-8:]) or "-"
            hist_placeholder.markdown(f"**Recent:** {recent_history}")

            word_placeholder.markdown(
                f"""
                <div style='background:#1a1a2e;
                            border-radius:10px;
                            padding:15px;
                            font-size:1.5rem;
                            font-weight:bold;
                            color:#00ff88;
                            min-height:60px;
                            letter-spacing:5px'>
                    {st.session_state.word or '...'}
                </div>
                """,
                unsafe_allow_html=True,
            )

            fps_placeholder.metric("FPS", fps)
            stat_placeholder.metric("Predictions", total_preds)

            # Screenshot download button
            if st.session_state.latest_frame is not None:
                _, screenshot_img = cv2.imencode('.jpg', st.session_state.latest_frame)
                screenshot_btn.download_button(
                    label="⬇️ Download",
                    data=screenshot_img.tobytes(),
                    file_name=f"screenshot_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                    key=f"download_screenshot_{int(time.time() * 1000)}"
                )

        cap.release()

    else:
        frame_placeholder.markdown(
            """
            <div style='background:#1a1a2e;
                        border-radius:15px;
                        height:400px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        border:2px dashed #333'>
                <div style='text-align:center;color:#94a3b8'>
                    <div style='font-size:3rem'>Camera idle</div>
                    <div style='font-size:1rem; margin-top:10px'>
                        Toggle the camera on when you are ready to start detection.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_image_prediction():
    """Upload image for prediction"""

    st.markdown("## 📸 Image Prediction")
    st.markdown("Upload an image of a hand sign to predict!")

    model, labels = load_model()

    if model is None:
        st.error("❌ Model not found! Run training first.")
        return

    # Upload
    uploaded = st.file_uploader(
        "Upload hand sign image",
        type = ['jpg', 'jpeg', 'png'],
        help = "Upload a clear image of your hand sign"
    )

    if uploaded:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📥 Uploaded Image")
            image = Image.open(uploaded)
            st.image(image,
                     caption="Your uploaded image",
                     use_column_width=True)

        # Process image
        img_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array,
                                    cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        # Resize
        img_bgr = cv2.resize(img_bgr, (224, 224))

        # Load MediaPipe for static image
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands_static = mp_hands.Hands(
            static_image_mode        = True,
            max_num_hands            = 1,
            min_detection_confidence = 0.5
        )

        # Extract landmarks
        rgb     = cv2.cvtColor(img_bgr,
                                cv2.COLOR_BGR2RGB)
        results = hands_static.process(rgb)
        hands_static.close()

        with col2:
            st.markdown("### 🎯 Prediction Result")

            if results.multi_hand_landmarks:
                # Extract landmarks
                hand_lms = results.multi_hand_landmarks[0]
                lm_list  = []
                for lm in hand_lms.landmark:
                    lm_list.extend([lm.x, lm.y, lm.z])

                lm_arr = np.array(
                    lm_list, dtype=np.float32
                )

                # Normalize
                lm_r  = lm_arr.reshape(21, 3)
                wrist = lm_r[0].copy()
                lm_r  = lm_r - wrist
                mx    = np.max(np.abs(lm_r))
                if mx > 0:
                    lm_r = lm_r / mx

                landmarks = lm_r.flatten()

                # Predict
                pred = model.predict(
                    landmarks.reshape(1,-1),
                    verbose=0
                )[0]

                # Top 3 predictions
                top3_idx  = np.argsort(pred)[::-1][:3]

                # Show top prediction
                best_letter = str(labels[top3_idx[0]])
                best_conf   = float(pred[top3_idx[0]])

                st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-letter">
                        {best_letter}
                    </div>
                    <div class="pred-conf">
                        Confidence: {best_conf*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📊 Top 3 Predictions")
                for i, idx in enumerate(top3_idx):
                    letter = str(labels[idx])
                    conf   = float(pred[idx])
                    st.progress(conf,
                        text=f"{i+1}. {letter} "
                             f"— {conf*100:.1f}%")

                # Display test accuracy
                status = check_project_status()
                if status['test_accuracy'] is not None:
                    st.markdown("---")
                    st.markdown("### 📈 Model Performance")
                    col_acc1, col_acc2 = st.columns(2)
                    with col_acc1:
                        st.metric("Test Accuracy", f"{status['test_accuracy']*100:.2f}%")

            else:
                st.warning("""
                ⚠️ **No hand detected in image!**

                Tips:
                - Make sure hand is clearly visible
                - Good lighting
                - Hand should fill most of the image
                """)


def page_status():
    """Project status page"""

    st.markdown("## 📊 Project Status")

    status = check_project_status()

    # Overall progress
    completed = sum([
        status['dataset'],
        status['X_npy'],
        status['model']
    ])

    st.progress(
        completed / 3,
        text=f"Overall Progress: {completed}/3 stages complete"
    )

    st.markdown("---")

    # Stage cards
    col1, col2 = st.columns(2)

    with col1:
        # Dataset status
        st.markdown("### 📁 Stage 3 — Dataset")
        if status['dataset']:
            st.success(
                f"✅ {status['gesture_count']} gesture folders\n\n"
                f"✅ {status['total_images']} total images"
            )
        else:
            st.error("❌ Dataset not found")
            st.code("python src/data_collection.py")

        st.markdown("### 🔧 Stage 4 — Preprocessing")
        if status['X_npy'] and status['y_npy']:
            X = np.load(os.path.join(BASE_DIR, "src", "data", "X.npy"))
            st.success(
                f"✅ X.npy — shape: {X.shape}\n\n"
                f"✅ y.npy — found\n\n"
                f"✅ labels.npy — found"
            )
        else:
            st.error("❌ .npy files not found")
            st.code("python src/data_preprocessing.py")

    with col2:
        # Model status
        st.markdown("### 🧠 Stage 5 — Model")
        if status['model']:
            size = os.path.getsize(MODEL_PATH) / 1024 / 1024
            accuracy_text = ""
            if status['test_accuracy'] is not None:
                accuracy_text = f"\n\n✅ Test Accuracy: {status['test_accuracy']*100:.2f}%"
            st.success(
                f"✅ sign_model.h5 found\n\n"
                f"✅ Size: {size:.1f} MB{accuracy_text}"
            )
        else:
            st.error("❌ Model not found")
            st.code("python src/model_training.py")

        # Training plot
        st.markdown("### 📈 Training Plot")
        if status['plot']:
            st.success("✅ training_plot.png found")
            img = Image.open(TRAINING_PLOT_PATH)
            st.image(img, use_column_width=True)
        else:
            st.warning("⚠️ No training plot yet")

    st.markdown("---")

    # Ready check
    if status['model'] and status['X_npy']:
        st.balloons()
        st.success("""
        🎉 **Everything is ready!**
        Go to **Live Prediction** to start!
        """)
    else:
        st.warning("""
        ⚠️ **Not ready yet!**
        Complete the missing stages above first.
        """)


def page_model_metrics():
    """Model Performance Metrics page"""

    st.markdown("## 📈 Model Performance Metrics")
    st.markdown("Detailed evaluation of the trained model on the test set")

    # Check if metric files exist
    cm_exists = os.path.exists(CONFUSION_MATRIX_PATH)
    roc_exists = os.path.exists(ROC_CURVE_PATH)
    pr_exists = os.path.exists(PR_CURVE_PATH)
    metrics_exists = os.path.exists(METRICS_SUMMARY_PATH)

    if not (cm_exists or roc_exists or pr_exists or metrics_exists):
        st.warning("""
        ⚠️ **Metrics not available yet!**
        
        To generate performance metrics, run the evaluation script:
        """)
        st.code("python src/model_evaluation.py", language="bash")
        st.info("""
        This will generate:
        - Confusion Matrix
        - ROC Curves
        - Precision-Recall Curves
        - Detailed Performance Report
        """)
        return

    st.markdown("---")

    # Tabs for different metric visualizations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Confusion Matrix", "📈 ROC Curves", "🎯 Precision-Recall", "📋 Metrics"]
    )

    # ── Tab 1: Confusion Matrix ──────────────────
    with tab1:
        st.markdown("### Confusion Matrix")
        st.markdown("""
        The confusion matrix shows the classification results. The diagonal represents
        correct predictions, while off-diagonal elements show misclassifications.
        """)

        if cm_exists:
            img = Image.open(CONFUSION_MATRIX_PATH)
            st.image(img, use_column_width=True,
                    caption="Heatmap showing actual vs predicted labels")
        else:
            st.error("❌ Confusion matrix not found.")
            st.info("Run: `python src/model_evaluation.py`")

    # ── Tab 2: ROC Curves ────────────────────────
    with tab2:
        st.markdown("### ROC Curves")
        st.markdown("""
        ROC (Receiver Operating Characteristic) curves show the tradeoff between
        True Positive Rate and False Positive Rate for each gesture class.
        - **Per-class ROC**: Individual curves for each gesture
        - **Micro-average ROC**: Overall model performance
        - **AUC**: Area Under the Curve (closer to 1 is better)
        """)

        if roc_exists:
            img = Image.open(ROC_CURVE_PATH)
            st.image(img, use_column_width=True,
                    caption="ROC curves for all gestures and micro-average")
        else:
            st.error("❌ ROC curve not found.")
            st.info("Run: `python src/model_evaluation.py`")

    # ── Tab 3: Precision-Recall Curves ───────────
    with tab3:
        st.markdown("### Precision-Recall Curves")
        st.markdown("""
        Precision-Recall (PR) curves show the relationship between precision and recall
        for each gesture class. Useful for imbalanced datasets.
        - **PR AUC**: Area Under the Precision-Recall Curve
        - Higher curves = better performance
        """)

        if pr_exists:
            img = Image.open(PR_CURVE_PATH)
            st.image(img, use_column_width=True,
                    caption="Precision-Recall curves for all gestures")
        else:
            st.error("❌ Precision-Recall curve not found.")
            st.info("Run: `python src/model_evaluation.py`")

    # ── Tab 4: Detailed Metrics ──────────────────
    with tab4:
        st.markdown("### Detailed Performance Report")

        if metrics_exists:
            with open(METRICS_SUMMARY_PATH, 'r') as f:
                metrics_text = f.read()

            st.text(metrics_text)

            # Download button
            st.download_button(
                label="📥 Download Metrics Report",
                data=metrics_text,
                file_name="performance_metrics.txt",
                mime="text/plain"
            )
        else:
            st.error("❌ Metrics report not found.")
            st.info("Run: `python src/model_evaluation.py`")

    st.markdown("---")

    # Summary section
    st.markdown("### 📝 Summary")

    status = check_project_status()
    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric_card(
            "Test Accuracy",
            f"{status['test_accuracy']*100:.2f}%" if status['test_accuracy'] else "N/A",
            "Overall model accuracy on test set"
        )

    with col2:
        render_metric_card(
            "Metrics",
            "✅ Available" if metrics_exists else "❌ Not Available",
            "Precision, Recall, F1-Score"
        )

    with col3:
        render_metric_card(
            "Visualizations",
            f"✅ {sum([cm_exists, roc_exists, pr_exists])}/3",
            "Confusion Matrix, ROC, Precision-Recall"
        )


def page_about():
    """About page"""

    st.markdown("## ℹ️ About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📌 Project Details
        | Field | Info |
        |-------|------|
        | **Project** | Sign Language Prediction |
        | **Student** | P. Bhanu Prakash |
        | **ID** | 23C11A6609 |
        | **College** | Anurag Engineering College |
        | **Dept** | CSE (AI & ML) |
        | **Guide** | Dr. G. John Babu |

        ### 🛠️ Tech Stack
        | Tool | Purpose |
        |------|---------|
        | Python | Core language |
        | OpenCV | Camera & image processing |
        | MediaPipe | Hand landmark detection |
        | TensorFlow | Neural network |
        | Streamlit | Web interface |
        | NumPy | Data processing |
        """)

    with col2:
        st.markdown("""
        ### 🧠 Model Architecture
```
        Input Layer    →  63 features
                          (21 landmarks × 3)
             ↓
        Dense(256)     →  ReLU activation
        BatchNorm + Dropout(0.3)
             ↓
        Dense(128)     →  ReLU activation
        BatchNorm + Dropout(0.3)
             ↓
        Dense(64)      →  ReLU activation
        BatchNorm + Dropout(0.2)
             ↓
        Output(26)     →  Softmax
                          (A to Z)
```

        ### 📊 Dataset
```
        Source   : Kaggle ASL Alphabet
        Images   : 87,000 total
        Classes  : 26 (A-Z)
        Per class: 3000 images
        Used     : 300 per class
```
        """)


# ══════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════

def main():

    # Render sidebar and get settings
    page, confidence, smooth = render_sidebar()

    # Route to correct page
    if page == "🏠 Home":
        page_home()

    elif page == "🎯 Live Prediction":
        page_live_prediction(confidence, smooth)

    elif page == "📸 Image Prediction":
        page_image_prediction()

    elif page == "📊 Project Status":
        page_status()

    elif page == "📈 Model Metrics":
        page_model_metrics()

    elif page == "ℹ️  About":
        page_about()


if __name__ == "__main__":
    main()
