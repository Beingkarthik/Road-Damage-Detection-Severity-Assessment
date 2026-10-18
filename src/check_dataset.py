from pathlib import Path

DATASET_DIR = Path("data/raw")

def main():
    print("RDD2022 Dataset Check")
    print("-" * 30)
    print("Dataset folder:", DATASET_DIR.resolve())
    print("Folder exists:", DATASET_DIR.exists())

if __name__ == "__main__":
    main()
