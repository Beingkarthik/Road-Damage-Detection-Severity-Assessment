# RoadLens — Road Damage Detection & Severity Assessment

**Deep Learning · Computer Vision · YOLOv8 · Streamlit**

RoadLens is a deep-learning-based road damage detection prototype that uses a trained YOLO object detection model to identify and localize potential road damage in images. It provides a web interface for uploading road photographs, reviewing predicted damage regions, inspecting confidence scores, and downloading annotated results.

**Live application:** [Launch RoadLens](https://road-damage-detection-severity-assessment-adpmrfbghjatfs8zck3z.streamlit.app/)

**Source code:** [GitHub repository](https://github.com/Beingkarthik/Road-Damage-Detection-Severity-Assessment)

> Research prototype: predictions are experimental and must not be treated as validated engineering assessments or used alone to make road-safety decisions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Deep Learning Approach](#deep-learning-approach)
- [Supported Damage Classes](#supported-damage-classes)
- [Application Features](#application-features)
- [Technology Stack](#technology-stack)
- [Dataset and Preparation](#dataset-and-preparation)
- [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Secure Model Checkpoint Setup](#secure-model-checkpoint-setup)
- [Deploy on Streamlit Community Cloud](#deploy-on-streamlit-community-cloud)
- [Limitations and Responsible Use](#limitations-and-responsible-use)
- [Future Improvements](#future-improvements)

---

## Project Overview

Road damage can affect vehicle safety, driving comfort, and road maintenance planning. Manually reviewing large collections of road images can be time-consuming. Computer vision provides a way to assist this process by identifying candidate damage regions in images.

RoadLens combines a trained YOLOv8 object detection model with a Streamlit application. The model predicts bounding boxes and class confidence scores for supported damage categories. The application presents the predictions in an interactive interface.

The project demonstrates an end-to-end deep learning workflow: dataset preparation, object detection, model inference, result visualization, and public web deployment.

## Problem Statement

Develop a computer vision prototype that detects and localizes selected categories of road damage from road images using deep learning, then presents the predictions through an accessible web application.

The intended goal is to support preliminary visual inspection—not to replace road engineers, field inspections, or validated pavement-condition assessment systems.

## Objectives

- Apply deep learning to road damage detection.
- Use YOLO-based object detection to predict damage classes and bounding boxes.
- Provide an interactive image inference interface.
- Display prediction confidence and annotated output images.
- Evaluate the model using object detection metrics.
- Deploy the application publicly while keeping the trained checkpoint outside the Git repository.
- Document model performance and limitations transparently.

## Deep Learning Approach

RoadLens uses **YOLOv8n**, a lightweight YOLO object detection model, through the Ultralytics framework.

The inference workflow is:

1. **Input:** A user uploads a road image.
2. **Preprocessing:** The image is normalized and passed to the detector.
3. **Deep learning inference:** The trained model predicts candidate damage bounding boxes, class IDs, and confidence scores.
4. **Confidence filtering:** Predictions below the selected confidence threshold are filtered out.
5. **Visualization:** The application renders the predicted regions on the image.
6. **Results:** The user reviews the detected classes and confidence scores and can download the annotated image.

YOLO predicts object locations and categories in a single detection pipeline. RoadLens uses the trained checkpoint for inference; it does not retrain the model when a user uploads an image.

## Supported Damage Classes

The current model supports five categories:

| Class ID | Category | Description |
|---|---|---|
| D00 | Longitudinal crack | Crack running mainly along the direction of travel |
| D10 | Transverse crack | Crack running mainly across the direction of travel |
| D20 | Alligator crack | Interconnected cracking resembling a mesh |
| D40 | Pothole | A localized cavity or broken area in the road surface |
| D44 | Other road damage | Other damage represented by this dataset label |

The application does not have a dedicated normal-road class. No detections do not prove that a road is damage-free.

## Application Features

- **Public web interface:** Access the deployed application in a browser.
- **Image upload:** Analyze supported image formats.
- **Adjustable confidence threshold:** Control which predictions are displayed.
- **Inference image size:** Configure the model's inference resolution.
- **Annotated predictions:** Inspect predicted damage regions and bounding boxes.
- **Detection details:** Review predicted categories, confidence scores, and bounding-box coordinates.
- **Download:** Save the annotated image for further review.
- **Private checkpoint retrieval:** Download the trained model from a private Hugging Face repository using a server-side secret.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Object detection | Ultralytics YOLOv8 |
| Computer vision | OpenCV, Pillow |
| Interactive web application | Streamlit |
| Model hosting | Hugging Face Hub |
| Source control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

## Dataset and Preparation

The project is based on the **RDD2022 (Road Damage Detection 2022)** dataset.

RDD2022 provides road images annotated for road damage detection. The project uses five selected categories: D00, D10, D20, D40, and D44.

Dataset preparation includes image and annotation organization, label conversion where required, validation of image-label pairs, and mapping the selected categories to the model's class IDs.

The final training run used a reconstructed Kaggle RDD2022 archive after a training-runtime reset. That archive contained images with VOC XML annotations, which were converted for the selected YOLO classes. Consequently, the reported metrics below apply to that reconstructed validation split—not to an independently verified evaluation on the original dataset's official test split.

The complete dataset is not included in this GitHub repository. Obtain and use the dataset according to its original access terms and licensing requirements.

## Model Evaluation

The available validation results for the trained checkpoint are:

| Metric | Validation result |
|---|---:|
| Precision | 0.3788 |
| Recall | 0.3263 |
| mAP@0.50 | 0.3494 |
| mAP@0.50:0.95 | 0.1855 |

### Per-class mAP@0.50

| Class | mAP@0.50 |
|---|---:|
| D00 — Longitudinal crack | 0.2483 |
| D10 — Transverse crack | 0.2437 |
| D20 — Alligator crack | 0.3954 |
| D40 — Pothole | 0.0159 |
| D44 — Other road damage | 0.0241 |

These results indicate that performance is limited and varies considerably by class. In particular, D40 and D44 have very low validation mAP. A successful application run confirms that the model can execute inference; it does not establish high detection accuracy.

**Metric definitions**

- **Precision:** The proportion of predicted detections that are correct under the evaluation matching criteria.
- **Recall:** The proportion of ground-truth objects successfully detected.
- **mAP@0.50:** Mean average precision at an intersection-over-union threshold of 0.50.
- **mAP@0.50:0.95:** Mean average precision averaged over IoU thresholds from 0.50 to 0.95.

Metrics depend on the dataset split, annotation quality, class distribution, and evaluation configuration. They should not be compared with results from other experiments unless the evaluation protocols are compatible.

## Project Structure

The main project components are organized as follows:

```text
Road-Damage-Detection-Severity-Assessment/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── completed_30_epochs_last.pt   # Local checkpoint; not committed
├── src/
│   ├── check_dataset.py
│   ├── main.py
│   └── train.py
├── notebooks/
├── results/
├── sample_test_images/
└── data/
    └── raw/
        └── RDD2022/                  # Local dataset; not committed
```

Some local folders may be absent from a fresh clone because datasets, model weights, training outputs, and other generated files are excluded from version control.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Beingkarthik/Road-Damage-Detection-Severity-Assessment.git
cd Road-Damage-Detection-Severity-Assessment
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Python 3.11 is a reasonable starting point for setting up this project. Install the dependencies declared in the repository:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure the trained checkpoint

Place the trained checkpoint at:

```text
models/completed_30_epochs_last.pt
```

Alternatively, configure the private Hugging Face model download as described below.

### 4. Launch RoadLens

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Secure Model Checkpoint Setup

The trained checkpoint is excluded from Git because it is a model artifact rather than application source code. The deployed application retrieves it from a private Hugging Face model repository.

Model repository:

[beingkarthik/roadlens-checkpoint](https://huggingface.co/beingkarthik/roadlens-checkpoint)

The current application expects the checkpoint file named:

`completed_30_epochs_last (1).pt`

### Configure Hugging Face access

1. Create a Hugging Face access token with the minimum read permissions needed for the private model repository.
2. In Streamlit Community Cloud, open the app's settings and locate **Secrets**.
3. Add the following TOML entry, substituting your own token locally:

```toml
HF_TOKEN = "YOUR_HUGGING_FACE_READ_TOKEN"
```

4. Save the secret and reboot the application if necessary.

**Security requirements**

- Never commit tokens, passwords, or credentials to GitHub.
- Never place the token directly in `app.py` or `requirements.txt`.
- Do not publish the private checkpoint in the application repository.
- If a token is exposed, revoke it and create a replacement.
- Give the token only the permissions necessary to read the model repository.

The public application can use the checkpoint without making the model repository public, provided its server-side token has the required permissions.

## Deploy on Streamlit Community Cloud

1. Push the application source code and `requirements.txt` to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create or open the app associated with this repository.
4. Select the `master` branch and `app.py` as the main file.
5. Add the `HF_TOKEN` secret using the instructions above.
6. Deploy or reboot the app.
7. Review the deployment logs if startup or checkpoint retrieval fails.
8. Upload a sample road image and verify that inference and result visualization work.

Live demo: [RoadLens — Road Damage Detection](https://road-damage-detection-severity-assessment-adpmrfbghjatfs8zck3z.streamlit.app/)

## Limitations and Responsible Use

- The model's measured validation performance is limited.
- D40 and D44 have particularly weak validation results.
- The evaluation reported here is not an official RDD2022 test-set evaluation.
- The model supports only the five listed damage categories.
- No detections do not establish that a road is safe or free from defects.
- Predictions can include false positives and false negatives.
- The application's severity-related interpretation is experimental and is not a validated engineering severity measurement.
- Performance may change with lighting, camera angle, image quality, road surface, and geographic conditions.
- Do not use this prototype as the sole basis for road-safety decisions, maintenance prioritization, or engineering certification.

## Future Improvements

- Evaluate against a properly separated, independently verified test set.
- Improve the representation and detection performance of underperforming classes.
- Compare YOLO model sizes and training configurations.
- Perform error analysis across lighting, weather, road type, and image quality.
- Add reproducible training configuration and experiment tracking.
- Establish a validated severity-assessment methodology before reporting engineering severity.
- Add automated regression tests and deployment health checks.
- Document dataset provenance, annotation quality, and model versioning.

---

## Acknowledgements

This project uses the RDD2022 road damage detection dataset and the Ultralytics YOLO framework.

- [RDD2022 dataset information](https://github.com/sekilab/RoadDamageDetector)
- [Ultralytics documentation](https://docs.ultralytics.com/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Hugging Face Hub documentation](https://huggingface.co/docs/huggingface_hub/)

---

**Project:** Road Damage Detection & Severity Assessment Using Deep Learning  
**Application:** RoadLens  
**Developer:** [Beingkarthik on GitHub](https://github.com/Beingkarthik)
