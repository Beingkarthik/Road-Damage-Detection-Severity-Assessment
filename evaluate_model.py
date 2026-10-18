from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "RDD2022" / "RDD_SPLIT" / "data.yaml"
MODEL = ROOT / "models" / "completed_30_epochs_last.pt"

def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")
    if not MODEL.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL}")

    model = YOLO(str(MODEL))
    print("Model classes:", model.names)
    print("Evaluating on the held-out test split. This may take time on CPU...")

    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=640,
        batch=4,
        device="cpu",
        plots=True,
        project=str(ROOT / "runs" / "evaluation"),
        name="test_results",
        exist_ok=True,
    )

    print("\n=== TEST METRICS ===")
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall:    {float(metrics.box.mr):.4f}")
    print(f"mAP50:     {float(metrics.box.map50):.4f}")
    print(f"mAP50-95:  {float(metrics.box.map):.4f}")
    print("\nEvaluation outputs:", ROOT / "runs" / "evaluation" / "test_results")

if __name__ == "__main__":
    main()
