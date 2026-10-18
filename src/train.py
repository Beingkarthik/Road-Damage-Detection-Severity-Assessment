from ultralytics import YOLO

DATA = "data/processed/data.yaml"
MODEL = "yolov8n.pt"

def main():
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=30,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=2,
        project="runs/road_damage",
        name="yolov8n_baseline",
        patience=10,
        plots=True,
        seed=42,
        deterministic=True,
        save=True,
        save_period=5,
        exist_ok=False,
    )

if __name__ == "__main__":
    main()