"""
LEONI MED — Industrial Connector Inspection using AI
=====================================================
A production-ready Streamlit application that loads a trained EfficientNetB0
binary classifier (best_model.keras) and predicts whether an electrical
connector terminal is Normal (piece conforme) or Defective (piece defect).

Run with:
    streamlit run app.py
"""

import json
import os
import threading

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from tensorflow.keras.applications.efficientnet import preprocess_input

# ------------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="LEONI MED | Connector Inspection",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------
MODEL_PATH = "best_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)
DEFECT_KEYWORDS = ("defect",)  # used to identify which class label means "defective"

COLOR_NORMAL = "#1DB954"     # green
COLOR_DEFECT = "#E63946"     # red
COLOR_NEUTRAL = "#4A4A4A"

# Live-feed settings
LIVE_INFER_EVERY_N_FRAMES = 5   # run the model every Nth frame to keep video smooth
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ------------------------------------------------------------------------------------
# Custom CSS — modern industrial look
# ------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Global font */
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }

        /* Main title block */
        .main-title {
            background: linear-gradient(90deg, #0F2027, #203A43, #2C5364);
            padding: 28px 32px;
            border-radius: 14px;
            margin-bottom: 24px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        }
        .main-title h1 {
            color: #FFFFFF;
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 4px;
            letter-spacing: 1px;
        }
        .main-title p {
            color: #C9D6DF;
            font-size: 1.05rem;
            margin: 0;
        }

        /* Result cards */
        .result-card {
            padding: 26px;
            border-radius: 14px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: 800;
            color: white;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            margin-top: 12px;
            margin-bottom: 12px;
        }
        .confidence-text {
            font-size: 1.05rem;
            font-weight: 500;
            color: #EAEAEA;
            margin-top: 6px;
        }

        /* Section headers */
        .section-header {
            font-size: 1.2rem;
            font-weight: 700;
            color: #203A43;
            border-left: 5px solid #2C5364;
            padding-left: 10px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #888888;
            font-size: 0.85rem;
            margin-top: 40px;
            padding-top: 14px;
            border-top: 1px solid #E0E0E0;
        }

        /* Legend badges */
        .legend-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            margin-right: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------------
# Cached resource loaders
# ------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_trained_model(path: str):
    if not os.path.exists(path):
        return None
    model = tf.keras.models.load_model(path)
    return model


@st.cache_data(show_spinner=False)
def load_class_names(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        mapping = json.load(f)
    # Ensure ordering by integer key: {"0": "...", "1": "..."}
    ordered = [mapping[str(i)] for i in range(len(mapping))]
    return ordered


def is_defect_label(label: str) -> bool:
    return any(keyword in label.lower() for keyword in DEFECT_KEYWORDS)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, convert to array and apply EfficientNet preprocessing."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def run_prediction(model, image: Image.Image, class_names):
    """Returns (predicted_label, confidence_percent, is_defect)."""
    processed = preprocess_image(image)
    prob = float(model.predict(processed, verbose=0).ravel()[0])

    # prob corresponds to P(class == 1) per class_names["1"]
    predicted_index = 1 if prob > 0.5 else 0
    predicted_label = class_names[predicted_index]
    confidence = prob if predicted_index == 1 else (1 - prob)
    confidence_pct = confidence * 100.0

    defect = is_defect_label(predicted_label)
    return predicted_label, confidence_pct, defect


def render_camera_selector(param_name: str):
    """Renders a dropdown of the browser's actual camera devices and returns
    the selected deviceId (or None for the default camera).

    Streamlit's Python side can't enumerate client-side hardware directly, so
    the selection is round-tripped through the page's URL query string: the
    JS below lists the real devices via navigator.mediaDevices, and writing a
    new query param triggers a rerun where Python reads it back out.
    """
    current = st.query_params.get(param_name, "")
    components.html(
        f"""
        <select id="{param_name}_select" style="padding:6px 10px;border-radius:8px;
            border:1px solid #ccc;font-size:0.95rem;min-width:280px;">
            <option value="">Detecting cameras...</option>
        </select>
        <script>
        (async function() {{
            const select = document.getElementById("{param_name}_select");
            try {{
                await navigator.mediaDevices.getUserMedia({{ video: true }});
                const devices = await navigator.mediaDevices.enumerateDevices();
                const cams = devices.filter(d => d.kind === "videoinput");
                select.innerHTML = "";
                const noneOpt = document.createElement("option");
                noneOpt.value = "";
                noneOpt.text = "Default camera";
                select.appendChild(noneOpt);
                cams.forEach((cam, i) => {{
                    const opt = document.createElement("option");
                    opt.value = cam.deviceId;
                    opt.text = cam.label || ("Camera " + (i + 1));
                    select.appendChild(opt);
                }});
                const current = "{current}";
                if (current) select.value = current;
            }} catch (e) {{
                select.innerHTML = '<option value="">Camera access denied</option>';
            }}
            select.addEventListener("change", function() {{
                const url = new URL(window.parent.location.href);
                if (this.value) {{
                    url.searchParams.set("{param_name}", this.value);
                }} else {{
                    url.searchParams.delete("{param_name}");
                }}
                window.parent.location.href = url.toString();
            }});
        }})();
        </script>
        """,
        height=50,
    )
    return current or None


class SnapshotVideoProcessor(VideoProcessorBase):
    """Keeps only the most recent frame in memory so a 'Capture Photo'
    button can grab a still image on demand (no continuous inference)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.latest_frame = img.copy()
        return frame


class LiveInspectionProcessor(VideoProcessorBase):
    """
    Runs the classifier on live webcam frames and draws the prediction
    directly on the video, so the operator sees a live label without
    clicking any capture button.
    """

    def __init__(self):
        self.frame_count = 0
        self.lock = threading.Lock()
        self.last_label = "Waiting..."
        self.last_confidence = 0.0
        self.last_defect = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1
        # Only run the (relatively expensive) model every N frames.
        if self.frame_count % LIVE_INFER_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            label, confidence_pct, defect = run_prediction(model, pil_image, class_names)
            with self.lock:
                self.last_label = label
                self.last_confidence = confidence_pct
                self.last_defect = defect

        with self.lock:
            label, confidence_pct, defect = self.last_label, self.last_confidence, self.last_defect

        # BGR colors for OpenCV drawing
        box_color = (57, 61, 230) if defect else (84, 185, 29)  # red / green in BGR
        display_text = "Defective Piece" if defect else "Normal Piece"

        overlay_height = 60
        cv2.rectangle(img, (0, 0), (img.shape[1], overlay_height), box_color, thickness=-1)
        cv2.putText(
            img,
            f"{display_text}  ({confidence_pct:.1f}%)",
            (12, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------------------------------------------------------------------------
# Load resources
# ------------------------------------------------------------------------------------
model = load_trained_model(MODEL_PATH)
class_names = load_class_names(CLASS_NAMES_PATH)

# ------------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔌 LEONI MED")
    st.markdown("### Project Description")
    st.write(
        "This application performs automated visual inspection of electrical "
        "connector terminals, classifying each piece as **Normal** or "
        "**Defective** using a deep learning model."
    )

    st.markdown("---")
    st.markdown("### 🧠 Model Information")
    st.write(
        """
        - **Architecture:** EfficientNetB0 (transfer learning)
        - **Input size:** 224 × 224 × 3
        - **Task:** Binary classification
        - **Framework:** TensorFlow / Keras
        """
    )

    if model is not None:
        st.success("Model loaded successfully ✅")
    else:
        st.error(f"Model file not found: `{MODEL_PATH}`")

    if class_names is not None:
        st.info(f"Classes: {', '.join(class_names)}")
    else:
        st.error(f"Class names file not found: `{CLASS_NAMES_PATH}`")

    st.markdown("---")
    st.markdown("### 🏷️ Prediction Legend")
    st.markdown(
        f"""
        <span class="legend-badge" style="background-color:{COLOR_NORMAL};">Normal Piece</span>
        <span class="legend-badge" style="background-color:{COLOR_DEFECT};">Defective Piece</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Adjust the decision threshold if needed for your production line.")

# ------------------------------------------------------------------------------------
# Main title
# ------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-title">
        <h1>LEONI MED</h1>
        <p>Industrial Connector Inspection using AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------------
# Guard: stop early if resources are missing
# ------------------------------------------------------------------------------------
if model is None or class_names is None:
    st.warning(
        "The application cannot run predictions until both `best_model.keras` "
        "and `class_names.json` are present in the working directory. "
        "Please run `train_model.ipynb` first to generate these artifacts."
    )
    st.stop()

# ------------------------------------------------------------------------------------
# Image input — upload or camera
# ------------------------------------------------------------------------------------
st.markdown('<div class="section-header">📷 Provide an Image</div>', unsafe_allow_html=True)

tab_upload, tab_camera, tab_live = st.tabs(
    ["📁 Upload Image", "📸 Camera Snapshot", "🔴 Live Real-Time Feed"]
)

input_image = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a connector terminal image",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

with tab_camera:
    st.write("Choose a camera below, then click **Capture Photo** to take a snapshot.")

    snapshot_device_id = render_camera_selector(param_name="snapshot_cam")
    snapshot_constraints = {"video": True, "audio": False}
    if snapshot_device_id:
        snapshot_constraints = {"video": {"deviceId": {"exact": snapshot_device_id}}, "audio": False}

    snapshot_ctx = webrtc_streamer(
        key="leoni-med-snapshot",
        video_processor_factory=SnapshotVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=snapshot_constraints,
    )

    capture_col, clear_col = st.columns([1, 1])
    with capture_col:
        if st.button("📸 Capture Photo", key="capture_snapshot_btn", use_container_width=True):
            if snapshot_ctx.video_processor:
                with snapshot_ctx.video_processor.lock:
                    frame = snapshot_ctx.video_processor.latest_frame
                if frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.session_state["captured_snapshot"] = Image.fromarray(rgb)
                    st.success("Photo captured below.")
                else:
                    st.warning("No frame available yet — start the camera and wait a moment.")
            else:
                st.warning("Start the camera first.")
    with clear_col:
        if st.button("🗑️ Clear Snapshot", key="clear_snapshot_btn", use_container_width=True):
            st.session_state.pop("captured_snapshot", None)

    if "captured_snapshot" in st.session_state:
        input_image = st.session_state["captured_snapshot"]

with tab_live:
    st.write(
        "Point the connector terminal at the camera. The prediction and "
        "confidence are overlaid on the video in real time — no need to "
        "click a capture button."
    )
    live_device_id = render_camera_selector(param_name="live_cam")
    live_constraints = {"video": True, "audio": False}
    if live_device_id:
        live_constraints = {"video": {"deviceId": {"exact": live_device_id}}, "audio": False}

    webrtc_streamer(
        key="leoni-med-live-inspection",
        video_processor_factory=LiveInspectionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=live_constraints,
    )
    st.caption(
        f"Inference runs every {LIVE_INFER_EVERY_N_FRAMES} frames to keep the video smooth. "
        "This live feed is independent of the upload/snapshot tabs above."
    )

# ------------------------------------------------------------------------------------
# Prediction & display
# ------------------------------------------------------------------------------------
if input_image is not None:
    col_image, col_result = st.columns([1, 1], gap="large")

    with col_image:
        st.markdown('<div class="section-header">🖼️ Uploaded Image</div>', unsafe_allow_html=True)
        st.image(input_image, use_container_width=True, caption="Input image (auto-resized to 224x224 for inference)")

    with st.spinner("Running inference..."):
        predicted_label, confidence_pct, defect = run_prediction(model, input_image, class_names)

    with col_result:
        st.markdown('<div class="section-header">🔍 Prediction Result</div>', unsafe_allow_html=True)

        if defect:
            display_text = "Defective Piece"
            bg_color = COLOR_DEFECT
            icon = "⚠️"
        else:
            display_text = "Normal Piece"
            bg_color = COLOR_NORMAL
            icon = "✅"

        st.markdown(
            f"""
            <div class="result-card" style="background-color:{bg_color};">
                {icon} Prediction:<br>{display_text}
                <div class="confidence-text">Confidence: {confidence_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(confidence_pct / 100.0, 0.0), 1.0))

        with st.expander("Raw model output details"):
            st.write(f"Predicted class label: `{predicted_label}`")
            st.write(f"Confidence: `{confidence_pct:.4f}%`")
            st.write(f"Class mapping used: `{class_names}`")
else:
    st.info("Upload an image or capture one with your camera to run an inspection.")

# ------------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Developed using TensorFlow + Streamlit — LEONI MED Industrial Vision System
    </div>
    """,
    unsafe_allow_html=True,
)
