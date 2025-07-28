# 🧠 Brain Tumor Detector (2D & 3D)

> *A deep learning project for brain tumor classification and segmentation using 2D and 3D medical images. Built as a learning experience and portfolio showcase — not intended for clinical use.*

---

## 📌 Overview

This project explores the use of computer vision in the medical field, specifically in detecting and segmenting brain tumors using deep learning models. It includes both 2D and 3D pipelines with an optional extension into tabular data-based ML.

The goal is not real-world deployment, but to deepen understanding in:
- Deep Learning (DL)
- Computer Vision
- Medical Image Processing
- Model Serving + Web Deployment
- Explainable AI (XAI)

---

## 🧠 Model Pipelines

### 🔹 2D Pipeline
1. **Binary Classifier:** Tumor / No Tumor
2. **Multiclass Classifier:** Glioma, Meningioma, Pituitary, No Tumor
3. **Segmentation Model:** Predicts tumor mask (if detected)

### 🔸 3D Pipeline
1. **Binary Classifier:** Tumor / No Tumor
2. **Segmentation Model:** 3D segmentation (NIfTI / NII files)

---

## 🌐 Planned Web Features

- **Dashboard UI** (user-friendly, for non-technical users)
- **Dark/Light Mode Toggle**
- **Smart Mode Selection:** Auto-routes files to 2D or 3D model based on file type
- **Chatbot Integration:** Using Gemini API to assist with questions on symptoms, diagnosis, etc.
- **Explainable AI (XAI):** Visual + textual explanations of model decisions
- **Optional ML Models:** For tabular medical data

---

## 📁 Project Structure (planned)
```
brain-tumor-detector/
├── notebooks/ # Training notebooks
├── src/ # Source code
├── saved_models/ # Exported weights
├── webapp/ # Frontend/backend code (TBD)
├── README.md
```
---

## ⚠️ Disclaimer

> This project is for educational purposes only.  
> The models and system are **not intended for real-world clinical use** and may not meet any regulatory standards.

---

## ✨ Status

✅ 2D Classifier & Segmenter (in progress: fine-tuning)  
🔄 3D Pipeline (starting soon)  
🛠️ Web integration planned (after models are finalized)

---

## 👨‍💻 Author

Made by **Caynaaa** (Tech Enthusiast & Informatics Student)  
Currently exploring AI, DL, and real-world model integration.

---
