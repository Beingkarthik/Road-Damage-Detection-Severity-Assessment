
from pathlib import Path
from datetime import datetime
from collections import Counter
import io
import hashlib

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageFile, UnidentifiedImageError
from ultralytics import YOLO

try:
    from streamlit_paste_button import paste_image_button
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATHS = [
    ROOT / "models" / "completed_30_epochs_last.pt",
    ROOT / "completed_30_epochs_last.pt",
]

Image.MAX_IMAGE_PIXELS = 80_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

st.set_page_config(
    page_title="RoadLens | Road Damage AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}
.hero {
    padding: 1.8rem 2rem;
    border-radius: 20px;
    background: linear-gradient(120deg, #102b46, #176b72);
    color: white;
    margin-bottom: 1.3rem;
}
.hero h1 { color: white; margin: 0; }
.hero p { color: #e7f4f4; margin: .45rem 0 0; }
div[data-testid="stMetric"] {
    border: 1px solid #e4e7ec;
    padding: 14px;
    border-radius: 14px;
}
.stButton button {
    border-radius: 10px;
    min-height: 42px;
}
</style>
<div class="hero">
    <h1>🛣️ RoadLens</h1>
    <p>AI-powered road damage detection and visual inspection</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 2. LOAD THE EXISTING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading your trained model...")
def load_model(path):
    return YOLO(str(path))


model_path = next(
    (p for p in MODEL_PATHS if p.is_file()),
    None,
)

if model_path is None:
    st.error(
        "Model file not found. Expected: "
        "models/completed_30_epochs_last.pt"
    )
    st.stop()

try:
    model = load_model(model_path)
except Exception as exc:
    st.error("The model could not be loaded.")
    st.exception(exc)
    st.stop()


# ============================================================
# 3. IMAGE DECODING
# ============================================================

def decode_image(image_bytes):
    """
    Decode an image supported by Pillow.
    Applies EXIF rotation, handles grayscale and transparency,
    and returns an RGB PIL image.
    """
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.seek(0)  # First frame for animated images.
        source.load()
        image = ImageOps.exif_transpose(source)

        # Composite transparent images over a white background.
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

        return image.copy()


# ============================================================
# 4. SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    st.header("⚙️ Inspection settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.01,
        max_value=0.90,
        value=0.10,
        step=0.01,
        help="Lower values return more candidate detections and may add false positives.",
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[320, 480, 640, 800, 960, 1024],
        value=960,
    )

    st.caption("Inference only. No training is performed.")

    with st.expander("Model diagnostics"):
        st.write("Checkpoint:", str(model_path))
        st.write("Task:", getattr(model, "task", "unknown"))
        st.write("Classes:", model.names)
        st.write("Device: CPU")
        st.write("Maximum detections: 300")


# ============================================================
# 5. IMAGE INPUT
# ============================================================

left, right = st.columns([1, 1], gap="large")

original = None
image_name = None
image_bytes = None

with left:
    st.subheader("1 · Add an image")

    if CLIPBOARD_AVAILABLE:
        try:
            paste_result = paste_image_button(
                label="📋 Paste image from clipboard",
                key="roadlens_clipboard",
                errors="raise",
            )

            if paste_result.image_data is not None:
                pasted = paste_result.image_data.convert("RGB")
                buffer = io.BytesIO()
                pasted.save(buffer, format="PNG")

                st.session_state["roadlens_image_bytes"] = (
                    buffer.getvalue()
                )
                st.session_state["roadlens_image_name"] = (
                    "pasted_image.png"
                )
                st.session_state.pop("roadlens_report", None)

        except Exception:
            st.info(
                "Clipboard paste is unavailable in this browser. "
                "Use the file picker instead."
            )
    else:
        st.info(
            "Clipboard support is not installed. "
            "You can still upload images."
        )

    # No extension filter: let Pillow validate supported image data.
    uploaded = st.file_uploader(
        "Upload an image in a supported format",
        type=None,
        key="roadlens_file",
        help=(
            "Try JPG, JPEG, PNG, WEBP, BMP, GIF, TIFF, or other "
            "formats supported by Pillow."
        ),
    )

    if uploaded is not None:
        try:
            image_bytes = uploaded.getvalue()
            original = decode_image(image_bytes)
            image_name = uploaded.name

            digest = hashlib.sha256(image_bytes).hexdigest()
            if st.session_state.get("roadlens_input_hash") != digest:
                st.session_state.pop("roadlens_report", None)
                st.session_state["roadlens_input_hash"] = digest

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            st.error(
                "This file could not be decoded as a supported image. "
                "Try saving it as PNG or JPEG."
            )
            st.caption(str(exc))

    elif "roadlens_image_bytes" in st.session_state:
        try:
            image_bytes = st.session_state["roadlens_image_bytes"]
            original = decode_image(image_bytes)
            image_name = st.session_state.get(
                "roadlens_image_name",
                "pasted_image.png",
            )
        except Exception as exc:
            st.error(f"Could not read the pasted image: {exc}")

    if original is not None:
        st.image(
            original,
            caption=f"Ready: {image_name}",
            use_container_width=True,
        )
        st.caption(
            f"Dimensions: {original.width} × {original.height} pixels"
        )
    else:
        st.info(
            "Paste an image or upload a picture to begin. "
            "Unsupported file types will be reported."
        )


# ============================================================
# 6. INFERENCE
# ============================================================

with right:
    st.subheader("2 · Detect road damage")

    if original is None:
        st.info("Add a valid image to start detection.")

    elif st.button(
        "🔎 Analyze road image",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Running inference with your existing model..."
            ):
                prediction = model.predict(
                    source=original,
                    conf=float(confidence),
                    imgsz=1280,
                    iou=0.50,
                    max_det=300,
                    device="cpu",
                    augment=True,
                    verbose=False,
                )[0]

                # Ultralytics plot() returns BGR; convert to RGB.
                annotated_array = prediction.plot()
                annotated = Image.fromarray(
                    annotated_array[:, :, ::-1].copy()
                )

                rows = []

                if prediction.boxes is not None:
                    for box in prediction.boxes:
                        class_id = int(box.cls.item())
                        confidence_score = float(box.conf.item())
                        coordinates = [
                            round(float(v), 1)
                            for v in box.xyxy[0].tolist()
                        ]

                        if isinstance(prediction.names, dict):
                            class_name = prediction.names.get(
                                class_id,
                                f"Class {class_id}",
                            )
                        else:
                            class_name = str(class_id)

                        rows.append({
                            "Damage class": class_name,
                            "Confidence (%)": round(
                                confidence_score * 100, 2
                            ),
                            "X1": coordinates[0],
                            "Y1": coordinates[1],
                            "X2": coordinates[2],
                            "Y2": coordinates[3],
                        })

                rows.sort(
                    key=lambda item: item["Confidence (%)"],
                    reverse=True,
                )

                st.session_state["roadlens_report"] = {
                    "image": annotated,
                    "rows": rows,
                    "name": image_name or "road_image",
                    "time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "threshold": confidence,
                    "image_size": image_size,
                    "original_size": (
                        original.width,
                        original.height,
                    ),
                }

        except Exception as exc:
            st.error("Inference failed. Error details:")
            st.exception(exc)


# ============================================================
# 7. RESULTS AND DOWNLOADS
# ============================================================

report = st.session_state.get("roadlens_report")

if report is not None:
    st.divider()
    st.subheader("📊 Inspection report")

    rows = report["rows"]
    highest = max(
        (row["Confidence (%)"] for row in rows),
        default=0.0,
    )

    metrics = st.columns(3)
    metrics[0].metric("Detected regions", len(rows))
    metrics[1].metric("Highest confidence", f"{highest:.1f}%")
    metrics[2].metric(
        "Confidence threshold",
        f'{report["threshold"]:.0%}',
    )

    if rows:
        counts = Counter(row["Damage class"] for row in rows)
        st.write("**Detections by class**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Damage class": name, "Count": count}
                    for name, count in counts.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    image_col, details_col = st.columns([1.1, 1], gap="large")

    with image_col:
        st.image(
            report["image"],
            caption="Annotated prediction",
            use_container_width=True,
        )

        image_buffer = io.BytesIO()
        report["image"].save(image_buffer, format="PNG")

        st.download_button(
            "⬇️ Download annotated image",
            data=image_buffer.getvalue(),
            file_name="roadlens_annotated.png",
            mime="image/png",
            use_container_width=True,
        )

    with details_col:
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download CSV report",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="roadlens_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning(
                "The model returned no detections at this threshold."
            )
            st.write(
                "This does not mean the image is invalid. "
                "Try a clearer road image or adjust the confidence threshold."
            )

    st.caption(
        f'Image: {report["name"]} · '
        f'Analyzed: {report["time"]} · '
        f'Inference size: {report["image_size"]}'
    )


# ============================================================
# 8. LIMITATIONS
# ============================================================

with st.expander("ℹ️ Supported images and model limitations"):
    st.write(
        "RoadLens accepts image files that Pillow can decode. "
        "This commonly includes JPEG, PNG, WEBP, BMP, GIF, and TIFF. "
        "Animated files use their first frame."
    )
    st.write(
        "SVG, PDF, HEIC/HEIF, camera RAW files, and corrupted images "
        "may require conversion or additional format-specific libraries."
    )
    st.write(
        "The model detects only patterns it learned from its training data. "
        "Lower confidence can increase detections but also false positives."
    )
    st.warning(
        "RoadLens is an AI inspection aid, not a certified road-safety "
        "assessment. A missed detection does not prove that a road is undamaged."
    )
