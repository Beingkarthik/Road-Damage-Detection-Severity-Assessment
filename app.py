from pathlib import Path
import os
import io

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

try:
    from streamlit_paste_button import paste_image_button
    PASTE_AVAILABLE = True
except ImportError:
    PASTE_AVAILABLE = False


# ---------------------------- Page configuration ----------------------------
st.set_page_config(
    page_title="RoadLens | Road Damage AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- Styling ------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1240px; padding-top: 1.6rem; padding-bottom: 3rem; }
      h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #14233b; }
      .hero {
        padding: 1.65rem 1.8rem; border-radius: 22px;
        background: linear-gradient(120deg, #101f38 0%, #173d62 65%, #176b78 100%);
        color: white; margin-bottom: 1.3rem;
        box-shadow: 0 12px 32px rgba(20,35,59,.12);
      }
      .hero h1 { color: white !important; margin: 0 0 .35rem 0; font-size: 2.15rem; }
      .hero p { color: #d8e8f4; margin: 0; font-size: 1rem; }
      .eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 700; color: #79e0d0; margin-bottom: .55rem; }
      .panel {
        background: white; border: 1px solid #e3e9f2; border-radius: 18px;
        padding: 1.15rem 1.2rem; box-shadow: 0 4px 18px rgba(26,44,70,.04);
      }
      .muted { color: #617189; font-size: .92rem; }
      .small-label { color: #617189; text-transform: uppercase; font-size: .72rem; letter-spacing: .09em; font-weight: 700; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #e3e9f2; padding: 1rem; border-radius: 14px; }
      div.stButton > button { border-radius: 10px; font-weight: 600; }
      [data-testid="stFileUploader"] { background: #fff; border: 1px dashed #9db0c7; border-radius: 14px; padding: .5rem; }
      .footer { color: #718096; font-size: .8rem; text-align: center; padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Utilities ----------------------------------
CLASS_NAMES = {
    0: "D00 · Longitudinal crack",
    1: "D10 · Transverse crack",
    2: "D20 · Alligator crack",
    3: "D40 · Pothole",
    4: "D44 · Other road damage",
}


def find_model_path() -> Path:
    project_dir = Path(__file__).resolve().parent

    # Use the local model when running on your computer.
    candidates = [
        project_dir / "models" / "completed_30_epochs_last.pt",
        project_dir / "models" / "completed_30_epochs_last (1).pt",
        project_dir / "models" / "best.pt",
        project_dir / "models" / "yolov8n_fresh_fast_best.pt",
    ]

    configured = os.getenv("ROAD_DAMAGE_MODEL")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return configured_path.resolve()

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Cloud fallback: retrieve the checkpoint from the private HF repository.
    token = os.getenv("HF_TOKEN", "")
    try:
        token = st.secrets.get("HF_TOKEN", token)
    except Exception:
        pass

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Configure it in Streamlit Cloud Secrets."
        )

    try:
        return Path(
            hf_hub_download(
                repo_id="beingkarthik/roadlens-checkpoint",
                filename="completed_30_epochs_last (1).pt",
                token=token,
                repo_type="model",
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Private model download failed. Check HF_TOKEN permissions, "
            "repository access, and the exact model filename. Error: "
            + str(exc)
        ) from exc


@st.cache_resource(show_spinner="Loading the trained YOLO model…")
def load_model(model_path: str):
    return YOLO(model_path)


def normalize_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        image = background
    return image


def run_detection(model, image: Image.Image, confidence: float, image_size: int):
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )
    result = results[0]
    plotted = Image.fromarray(result.plot()[..., ::-1])
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            name = CLASS_NAMES.get(class_id, result.names.get(class_id, f"Class {class_id}"))
            detections.append(
                {"Class": name, "Confidence": f"{score:.1%}", "Bounding box (x1,y1,x2,y2)": str(xyxy)}
            )
    return plotted, detections


# -------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("## 🛣️ RoadLens")
    st.caption("Road damage detection workspace")
    st.divider()
    st.markdown("### Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Higher values show fewer, more confident predictions.",
    )
    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640, 768, 960],
        value=640,
        help="Larger sizes may improve small-object visibility but take longer.",
    )
    st.divider()
    st.markdown("### Supported classes")
    for class_name in CLASS_NAMES.values():
        st.caption("• " + class_name)
    st.divider()
    st.caption("Local inference · YOLOv8 · Experimental research prototype")


# --------------------------------- Header ------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer vision · AI inspection</div>
      <h1>Road Damage Detection</h1>
      <p>Upload or paste a road image, run your trained model, and review predicted damage regions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_path = find_model_path()
if not model_path.is_file():
    st.error(
        f"Model checkpoint not found at `{model_path}`. "
        "Place `completed_30_epochs_last.pt` in the project's `models` folder, "
        "or set the ROAD_DAMAGE_MODEL environment variable."
    )
    st.info("Your app interface can still be viewed, but detection requires the trained checkpoint.")
    st.stop()

# ------------------------------ Image input ---------------------------------
st.markdown("### 1 · Add a road image")
st.markdown(
    '<p class="muted">Choose a local image or paste an image copied to your clipboard. '
    'For clipboard pasting, Chrome or Edge usually works best.</p>',
    unsafe_allow_html=True,
)

upload_col, paste_col = st.columns([1.25, 1], gap="large")
uploaded_file = None
pasted_image = None

with upload_col:
    with st.container(border=True):
        st.markdown("**📁 Upload from your device**")
        uploaded_file = st.file_uploader(
            "Choose a road image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP, BMP",
            label_visibility="collapsed",
            key="road_image_upload",
        )
        st.caption("JPG · PNG · WEBP · BMP")

with paste_col:
    with st.container(border=True):
        st.markdown("**📋 Paste from clipboard**")
        st.caption("First copy an image, then click the button below.")
        if PASTE_AVAILABLE:
            try:
                paste_result = paste_image_button(
                    "Paste copied image",
                    key="road_damage_clipboard_paste",
                    background_color="#176b78",
                    hover_background_color="#12545e",
                    text_color="#ffffff",
                    errors="ignore",
                )
                if paste_result and paste_result.image_data is not None:
                    pasted_image = normalize_image(paste_result.image_data)
                    st.success("Image pasted successfully.")
            except Exception:
                st.warning(
                    "Clipboard access is unavailable in this browser session. "
                    "Use the upload option instead."
                )
        else:
            st.warning("Clipboard feature is not installed yet. Run the installation command shown below.")
        st.caption("Clipboard support depends on browser permissions and may not work in every browser.")

if not PASTE_AVAILABLE:
    with st.expander("Enable clipboard paste"):
        st.code("python -m pip install streamlit-paste-button", language="bash")
        st.caption("Then add `streamlit-paste-button` to requirements.txt and restart Streamlit.")

# Persist pasted image across Streamlit reruns
if pasted_image is not None:
    st.session_state["road_damage_pasted_image"] = pasted_image

selected_image = None
source_label = None
if uploaded_file is not None:
    try:
        selected_image = normalize_image(Image.open(uploaded_file))
        source_label = uploaded_file.name
    except Exception:
        st.error("The selected file could not be read as an image.")
elif st.session_state.get("road_damage_pasted_image") is not None:
    selected_image = st.session_state["road_damage_pasted_image"]
    source_label = "Pasted clipboard image"

if selected_image is not None:
    st.markdown("### 2 · Review image")
    preview_col, info_col = st.columns([1.35, 1], gap="large")
    with preview_col:
        st.image(selected_image, caption=source_label, use_container_width=True)
    with info_col:
        st.markdown(
            """
            <div class="panel">
              <div class="small-label">Image ready</div>
              <h3 style="margin:.45rem 0 .6rem 0;">Ready for analysis</h3>
              <p class="muted">The image will be processed locally by the trained model. It is not automatically uploaded to a cloud service by this app.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        run_clicked = st.button(
            "🔍  Detect road damage",
            type="primary",
            use_container_width=True,
        )
        if st.button("Clear selected image", use_container_width=True):
            st.session_state.pop("road_damage_pasted_image", None)
            st.rerun()

    if run_clicked:
        try:
            with st.spinner("Analyzing road image…"):
                model = load_model(str(model_path))
                annotated_image, detections = run_detection(
                    model, selected_image, confidence, image_size
                )

            st.markdown("### 3 · Detection results")
            result_col, summary_col = st.columns([1.35, 1], gap="large")
            with result_col:
                st.image(annotated_image, caption="Model predictions", use_container_width=True)
            with summary_col:
                st.metric("Detected regions", len(detections))
                st.metric("Confidence threshold", f"{confidence:.0%}")
                if detections:
                    st.success(f"Found {len(detections)} candidate road-damage region(s).")
                    st.dataframe(detections, use_container_width=True, hide_index=True)
                else:
                    st.info("No regions passed the current confidence threshold. This does not prove the road is damage-free.")
            output = io.BytesIO()
            annotated_image.save(output, format="PNG")
            st.download_button(
                "⬇️ Download annotated image",
                data=output.getvalue(),
                file_name="road_damage_predictions.png",
                mime="image/png",
            )
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.caption("Confirm the model checkpoint is valid and that the project dependencies are installed.")

st.divider()
st.markdown("### About this prototype")
st.warning(
    "Research prototype only. Predictions may be missed or incorrect. "
    "The model's validation performance was limited, especially for some damage classes. "
    "Do not use these predictions alone for road-safety decisions or engineering severity ratings."
)
st.markdown(
    '<div class="footer">RoadLens · Road Damage Detection & Severity Assessment · Built with Streamlit and YOLO</div>',
    unsafe_allow_html=True,
)
