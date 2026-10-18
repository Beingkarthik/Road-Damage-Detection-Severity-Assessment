# 🛣️ RoadLens — Road Damage Detection & Severity Assessment Using Deep Learning

**An AI-powered computer vision application for detecting road-surface damage from images using a trained YOLO object-detection model.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://road-damage-detection-severity-assessment-vgzqefg8dabed4bkvx8q.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Project%20Page-yellow?logo=huggingface)](https://huggingface.co/spaces/beingkarthik/roadlens-road-damage)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

## 🚀 Live Application

**Try RoadLens:** [Launch the live application](https://road-damage-detection-severity-assessment-vgzqefg8dabed4bkvx8q.streamlit.app/)

**Project website:** [Visit RoadLens on Hugging Face](https://huggingface.co/spaces/beingkarthik/roadlens-road-damage)

RoadLens provides a web interface for uploading road images, running inference with the existing trained model, and reviewing the resulting predictions.

## 📌 Project Overview

Road damage can affect road quality, vehicle safety, and maintenance planning. Manually inspecting large road networks is time-consuming and can be difficult to scale.

RoadLens explores how deep learning and computer vision can assist visual road inspection by detecting damage patterns in road images.

The application uses a trained YOLO model to identify configured road-damage classes and display predicted locations, class labels, and confidence scores.

### Problem Statement

Manual road inspection requires time and human effort, while road damage can vary in shape, size, and appearance. The objective of this project is to develop an image-based detection system that can assist inspectors by automatically identifying potential road damage in uploaded images.

### Project Objectives

- Detect configured road-damage patterns in images.
- Visualize predicted damage locations using bounding boxes.
- Display predicted classes and confidence scores.
- Provide an interactive web interface for model inference.
- Make the trained model accessible through a deployed application.

## ✨ Features

- **Image upload:** Upload supported road images through the web interface.
- **AI-powered detection:** Run inference using the trained YOLO model.
- **Bounding-box visualization:** Inspect the locations of predicted damage.
- **Adjustable inference settings:** Configure the confidence threshold and inference image size.
- **Prediction inspection:** Review available labels and confidence values.
- **Report and download features:** Use the output options available in the application.
- **Online deployment:** Access the application through Streamlit Community Cloud.
- **Project landing page:** Explore project information through Hugging Face Spaces.

## 🧠 Model Information

The application uses an existing trained YOLO checkpoint:

`models/completed_30_epochs_last.pt`

The deployed model is configured with the following class labels:

| Class ID | Label |
|---|---|
| 0 | D00 |
| 1 | D10 |
| 2 | D20 |
| 3 | D40 |
| 4 | D44 |

The exact class descriptions should be interpreted according to the label mapping used when preparing the training dataset. The model's predictions depend on its training data, image quality, viewpoint, lighting, and inference settings.

**No retraining is performed when a user uploads an image.** The application uses the existing trained checkpoint for inference.

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and inference logic |
| YOLO / Ultralytics | Object detection |
| PyTorch | Deep-learning model execution |
| Streamlit | Interactive web interface |
| OpenCV | Computer vision and image processing |
| Pillow | Image loading and handling |
| Pandas | Tabular data and report handling |
| Git and GitHub | Version control and source hosting |
| Hugging Face Spaces | Static project landing page |
| Streamlit Community Cloud | Live application deployment |

## 🏗️ Application Workflow

```text
Road Image
    |
    v
Image Upload
    |
    v
Image Validation and Preprocessing
    |
    v
Trained YOLO Model
    |
    v
Road-Damage Predictions
    |
    v
Bounding Boxes + Labels + Confidence Scores
    |
    v
Results and Available Downloads
```

## 🖥️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Beingkarthik/Road-Damage-Detection-Severity-Assessment.git
cd Road-Damage-Detection-Severity-Assessment
```

### 2. Create a virtual environment

**Windows:**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the trained model

Ensure the model checkpoint is present at:

```text
models/completed_30_epochs_last.pt
```

### 5. Start the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, typically `http://localhost:8501`.

> **Note:** Local execution requires compatible Python packages and the trained model file. The Python version supported by the installed dependencies may differ from the version used by the cloud deployment.

## 📁 Project Structure

```text
Road-Damage-Detection-Severity-Assessment/
│
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
│
├── models/
│   └── completed_30_epochs_last.pt
│
├── data/
│   └── raw/
│
├── notebooks/
├── src/
│   ├── main.py
│   ├── train.py
│   ├── train_improved.py
│   └── check_dataset.py
│
├── sample_test_images/
├── runs/
└── evaluate_model.py
```

*This is a representative project layout; optional directories and supporting files may vary.*

## 🧪 Testing and Evaluation

The application can be tested using supported road images, including sample images from the project's test-image collection.

Recommended checks include:

- Confirming that the model loads successfully.
- Uploading a valid road image.
- Checking predicted labels and bounding boxes.
- Comparing predictions on known sample images.
- Testing confidence-threshold adjustments.
- Verifying available report and download features.
- Testing the deployed application after changes.

Quantitative performance metrics such as precision, recall, mAP, and inference time should be added only after they have been measured and verified on an appropriate evaluation dataset.

## ⚠️ Limitations

- Predictions are limited to patterns represented in the training data.
- Poor lighting, unusual camera angles, low image quality, and unfamiliar road surfaces can affect results.
- False positives and missed detections are possible.
- A low confidence threshold may increase detections as well as false positives.
- The model's outputs should not be treated as a certified assessment of road safety or damage severity.
- A missing detection does not establish that a road is undamaged.

RoadLens is a prototype intended to assist visual inspection and demonstrate deep-learning-based road damage detection.

## 👥 Contributors

This project was developed collaboratively by:

1. **Nanavath Karthik**
2. **Bendi Kushwanth**
3. **Shruthika**

## 🔮 Future Improvements

- Evaluate the model on a held-out dataset and document verified metrics.
- Improve generalization across different road and weather conditions.
- Investigate additional damage categories and severity estimation.
- Add more comprehensive batch-image processing and reporting.
- Improve testing, documentation, and deployment monitoring.

## 📄 License

No license is specified here. Before distributing or reusing this project, add a `LICENSE` file that reflects the contributors' chosen licensing terms and any applicable dataset or model restrictions.

---

**RoadLens — Deep learning for smarter road inspection.**
