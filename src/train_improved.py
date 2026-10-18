from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "data.yaml"
MODEL = ROOT / "yolov8n.pt"


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")
    if not MODEL.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {MODEL}")

    model = YOLO(str(MODEL))
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=2,
        project=str(ROOT / "runs" / "road_damage"),
        name="yolov8n_improved",
        patience=12,
        plots=True,
        seed=42,
        deterministic=True,
        save=True,
        save_period=5,
        exist_ok=True,
    )

    print("Training finished. Inspect the run folder for results.csv and weights/best.pt.")


if __name__ == "__main__":
    main()
