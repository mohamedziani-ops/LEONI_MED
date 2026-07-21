"""
LEONI MED — Inspection de connecteurs électriques
Application Streamlit : classification Normal / Défectueux

Modes disponibles :
  1. Importer une image
  2. Prendre une photo (avec choix de la caméra)
  3. Classification en direct (flux vidéo temps réel)

Fichiers requis dans le même dossier que app.py :
  - best_model.keras
  - class_names.json
"""

import json
import os
import time

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from streamlit_javascript import st_javascript

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "best_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="LEONI MED - Inspection Connecteurs",
    page_icon="🔌",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Chargement du modèle et des classes (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable : '{MODEL_PATH}'. Placez-le à côté de app.py.")
        st.stop()
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"Fichier de classes introuvable : '{CLASS_NAMES_PATH}'.")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_indices = json.load(f)  # {"0": "piece conforme", "1": "piece defect"}
    return model, class_indices


model, class_indices = load_model_and_classes()


def predict(image_rgb: np.ndarray):
    """Prend une image RGB (H, W, 3) uint8 et retourne (label, confidence)."""
    resized = cv2.resize(image_rgb, IMG_SIZE)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    batch = preprocess_input(batch)
    prob = float(model.predict(batch, verbose=0)[0][0])

    if prob > 0.5:
        label = class_indices["1"]
        confidence = prob
    else:
        label = class_indices["0"]
        confidence = 1 - prob

    return label, confidence


def display_result(label: str, confidence: float):
    is_defect = "defect" in label.lower()
    if is_defect:
        st.error(f"⚠️ **{label.upper()}** — confiance : {confidence * 100:.1f}%")
    else:
        st.success(f"✅ **{label.upper()}** — confiance : {confidence * 100:.1f}%")


# ---------------------------------------------------------------------------
# Sélection de la caméra (liste des périphériques vidéo du navigateur)
# ---------------------------------------------------------------------------
def get_camera_devices():
    js_code = """
    await new Promise((resolve) => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(() => navigator.mediaDevices.enumerateDevices())
            .then((devices) => {
                const cams = devices
                    .filter((d) => d.kind === "videoinput")
                    .map((d, i) => ({
                        label: d.label || ("Caméra " + (i + 1)),
                        deviceId: d.deviceId,
                    }));
                resolve(cams);
            })
            .catch(() => resolve([]));
    });
    """
    result = st_javascript(js_code)
    if isinstance(result, list) and len(result) > 0:
        return result
    return []


def camera_selector(widget_key: str):
    devices = get_camera_devices()
    if devices:
        labels = [d["label"] for d in devices]
        choice = st.selectbox("📷 Choisir la caméra", labels, key=f"cam_{widget_key}")
        device_id = next(d["deviceId"] for d in devices if d["label"] == choice)
        return {"video": {"deviceId": {"exact": device_id}}, "audio": False}
    else:
        st.caption(
            "Impossible de lister les caméras (autorisez l'accès caméra dans le "
            "navigateur). La caméra par défaut sera utilisée."
        )
        return {"video": True, "audio": False}


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("🔌 LEONI MED — Inspection de connecteurs")
st.caption("Classification : Pièce conforme vs Pièce défectueuse")

mode = st.radio(
    "Choisissez un mode :",
    ["📁 Importer une image", "📷 Prendre une photo", "🎥 Classification en direct"],
    horizontal=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Mode 1 : Importer une image
# ---------------------------------------------------------------------------
if mode == "📁 Importer une image":
    uploaded_file = st.file_uploader(
        "Importer une image du connecteur", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image importée", use_container_width=True)

        if st.button("🔍 Analyser", type="primary"):
            with st.spinner("Analyse en cours..."):
                label, confidence = predict(np.array(image))
            display_result(label, confidence)

# ---------------------------------------------------------------------------
# Mode 2 : Prendre une photo (avec sélection de caméra)
# ---------------------------------------------------------------------------
elif mode == "📷 Prendre une photo":
    constraints = camera_selector("photo")

    class SnapshotProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame = img
            return frame

    ctx = webrtc_streamer(
        key="photo-capture",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints=constraints,
        video_processor_factory=SnapshotProcessor,
        async_processing=True,
    )

    if st.button("📸 Capturer la photo", type="primary"):
        if ctx.video_processor and ctx.video_processor.frame is not None:
            bgr = ctx.video_processor.frame
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            st.session_state["captured_frame"] = rgb
        else:
            st.warning("Démarrez la caméra puis attendez l'image avant de capturer.")

    if "captured_frame" in st.session_state:
        rgb = st.session_state["captured_frame"]
        st.image(rgb, caption="Photo capturée", use_container_width=True)
        with st.spinner("Analyse en cours..."):
            label, confidence = predict(rgb)
        display_result(label, confidence)

# ---------------------------------------------------------------------------
# Mode 3 : Classification en direct (temps réel)
# ---------------------------------------------------------------------------
else:
    constraints = camera_selector("live")
    st.caption(
        "Le résultat s'affiche directement sur le flux vidéo. "
        "L'inférence tourne toutes les quelques images pour rester fluide."
    )

    class LiveClassifier(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.skip = 5  # exécute le modèle 1 image sur 5
            self.last_label = "..."
            self.last_conf = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            self.frame_count += 1
            if self.frame_count % self.skip == 0:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    self.last_label, self.last_conf = predict(rgb)
                except Exception:
                    pass

            is_defect = "defect" in self.last_label.lower()
            color = (0, 0, 255) if is_defect else (0, 200, 0)  # BGR
            text = f"{self.last_label.upper()} ({self.last_conf * 100:.0f}%)"

            cv2.rectangle(img, (0, 0), (img.shape[1], 40), color, -1)
            cv2.putText(
                img, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live-classification",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints=constraints,
        video_processor_factory=LiveClassifier,
        async_processing=True,
    )
