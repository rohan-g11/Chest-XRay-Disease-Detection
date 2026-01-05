# Chest X-Ray Disease Detection using Deep Learning

## 📌 Overview
This project implements a **multi-label chest X-ray disease classification system** using **Deep Learning and Transfer Learning**.  
The model is trained on the **NIH Chest X-Ray dataset** from Kaggle and is capable of detecting multiple thoracic diseases from a single X-ray image.

This is my **first deep learning project**, developed as part of my learning journey in **Artificial Intelligence & Data Science**.

---

## 🧠 Key Features
- Multi-label disease classification
- Patient-level train/validation split to prevent data leakage
- Advanced preprocessing using CLAHE & histogram equalization
- Data augmentation using Albumentations and Keras
- Transfer Learning with **DenseNet121**
- Fine-tuning for improved performance
- Model explainability using **Grad-CAM heatmaps and bounding boxes**

---

## 📊 Dataset
- **Source:** NIH Chest X-Ray Dataset (Kaggle)
- **Images:** Chest X-ray PNG images
- **Labels:** Multi-label disease annotations
- **CSV File:** `Data_Entry_2017.csv`

---

## ⚙️ Model Architecture
- Base Model: **DenseNet121 (ImageNet weights)**
- Global Average Pooling
- Batch Normalization
- Fully Connected Layers with Dropout
- Output Layer: Sigmoid activation for multi-label classification

---

## 📈 Training Strategy
- Binary Cross-Entropy loss with label smoothing
- Adam optimizer with learning rate scheduling
- Early stopping and model checkpointing
- Two-phase training:
  - Warm-up (frozen base model)
  - Fine-tuning (partial unfreezing)

---

## 🔍 Explainability (Grad-CAM)
Grad-CAM is used to visualize regions of interest in X-ray images, helping interpret model predictions by highlighting disease-relevant areas.

---

## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- NumPy & Pandas
- OpenCV
- Albumentations
- Matplotlib & Seaborn
- Jupyter Notebook

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
jupyter notebook chest_xray_multilabel_densenet121.ipynb
