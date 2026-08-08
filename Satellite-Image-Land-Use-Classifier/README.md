# Satellite Image Land-Use Classifier & Temporal Change Detector

A computer vision project for satellite image land-use classification and temporal change detection using CNN, ResNet18, transfer learning, and embedding-based cosine similarity.

## Project Overview

This project focuses on classifying satellite images into different land-use categories and detecting changes between satellite images captured at different times.

The project compares a baseline CNN model with a ResNet18 transfer-learning approach. ResNet18 provides improved classification performance, while deep image embeddings and cosine similarity are used for temporal change detection.

## Objectives

- Classify satellite images into 10 land-use categories.
- Build and evaluate a baseline CNN model.
- Improve classification performance using ResNet18 and transfer learning.
- Generate deep feature embeddings from satellite images.
- Detect temporal changes using cosine similarity.
- Evaluate model performance using accuracy, Macro F1-score, confusion matrix, and ROC-AUC.

## Dataset

### EuroSAT Dataset

- Images: 27,000
- Classes: 10
- Image type: Satellite imagery
- Land-use categories include:
  - AnnualCrop
  - Forest
  - HerbaceousVegetation
  - Highway
  - Industrial
  - Pasture
  - PermanentCrop
  - Residential
  - River
  - SeaLake

## Methodology

### 1. Data Preparation
- Loaded the satellite image dataset.
- Applied image preprocessing and transformations.
- Prepared the data for model training and evaluation.

### 2. Baseline CNN
A custom CNN model was developed as a baseline for land-use classification.

**Baseline Macro F1:** 84.25%

### 3. ResNet18 Transfer Learning
ResNet18 was used with transfer learning to improve image classification performance.

**ResNet18 Results:**
- Macro F1: 96.48%
- Test Accuracy: 97%

### 4. Feature Embeddings
The trained ResNet18 model was used to generate 512-dimensional image embeddings.

These embeddings represent high-level visual features from satellite images.

### 5. Temporal Change Detection
Cosine similarity was applied to compare image embeddings and identify differences between satellite images.

A lower similarity indicates greater visual change between the compared images.

## Key Results

| Metric | Result |
|---|---:|
| EuroSAT Dataset | 27,000 images |
| Number of Classes | 10 |
| Baseline CNN Macro F1 | 84.25% |
| ResNet18 Macro F1 | 96.48% |
| ResNet18 Test Accuracy | 97% |
| UC Merced Holdout | 2,100 images |
| Embedding Dimension | 512 |
| Change Detection | Cosine Similarity |

## Technologies

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit

## Project Structure

```text
Satellite-Image-Land-Use-Classifier/
│
├── README.md
├── app.py
├── requirements.txt
├── Satellite_Image_Land_Use_Classifier_Temporal_Change_Detection.ipynb
└── resnet18_eurosat.pt
