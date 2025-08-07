# 🧠 Brain Tumor Detector (2D & 3D)

> *A deep learning project for brain tumor classification and segmentation using 2D and 3D medical images. Built as a learning experience and portfolio showcase — not intended for clinical use.*

---

## 📌 Overview

> **⚠️ Update:** This project originally started using pure PyTorch for its 2D models. However, midway through development, I decided to transition to **PyTorch Lightning** to adopt a more modular, scalable, and production-ready structure — a reflection of real-world practices. The original PyTorch notebooks are preserved in the `Archive/` folder as a learning reference.

This project explores the use of computer vision in the medical field, specifically in detecting and segmenting brain tumors using deep learning models. It includes both 2D and 3D pipelines with an optional extension into tabular data-based ML.

The goal is not real-world deployment, but to deepen understanding in:
- Deep Learning (DL)
- Computer Vision
- Medical Image Processing
- Model Serving + Web Deployment
- Explainable AI (XAI)

---

## 🧠 Model Pipelines

### 🔹 2D Pipeline - (.JPG, .PNG)
1. **Binary Classifier:** Tumor / No Tumor
2. **Multiclass Classifier:** Glioma, Meningioma, Pituitary, No Tumor
3. **Segmentation Model:** Predicts tumor mask (if detected)

### 🔸 3D Pipeline - (.NIfTI / .NII files)
1. **Binary Classifier:** Tumor / No Tumor
2. **Segmentation Model:** 3D segmentation 

---

## 🌐 Planned Web Features

- **Dashboard UI** (user-friendly, for non-technical users)
- **Dark/Light Mode Toggle**
- **Smart Mode Selection:** Auto-routes files to 2D or 3D model based on file type
- **Chatbot Integration:** Using Gemini API to assist with questions on symptoms, diagnosis, etc.
- **Explainable AI (XAI):** Visual + textual explanations of model decisions
- **Optional ML Models:** For tabular medical data

---

## 📁 Project Structure (updated)

As part of this ongoing learning project, the codebase has been gradually transitioning to a more modular and scalable format using **PyTorch Lightning**, which better reflects modern practices and real-world applications. Earlier experiments and baseline models built using raw PyTorch have been retained under `Archives/` as a reference and learning trace.

> **Note:** The example below expands the internal structure of `Models/Classifier_Binary/`. Other models will follow a similar modular design.
```
brain-tumor-ai/
├── Archives/ # Legacy notebooks (PyTorch-based)
│
├── Models/ # Main folder for Lightning-based models
│ └── Classifier_Binary/ # Binary classification model
│ ├── module.py # LightningModule: defines model, loss, optimizer
│ ├── datamodule.py # DataModule: handles loading & preprocessing
│ ├── transform.py # Image augmentations (Albumentations)
│ ├── helper.py # Utility functions (e.g., seed, device)
│ ├── config.py # Config: model name, hyperparams, data paths
│ ├── callbacks.py # Callbacks (checkpoint, early stopping, etc.)
│ ├── runner.ipynb # Notebook for training/evaluation (Colab-ready)
│ └── checkpoint/ # Saved model weights (.ckpt, .pt if exported)
│
├── webapp/ # Web interface code (TBD)
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

## 🛠️ Development Notes

- The project has transitioned from raw PyTorch to PyTorch Lightning for a more modular and scalable design.
- **[Correction]** The initial binary classification model incorrectly used `CrossEntropyLoss`. This has been fixed and replaced with `BCEWithLogitsLoss` which is suitable for binary tasks.
---

## 👨‍💻 Author

Made by **Caynaaa** (Tech Enthusiast & Informatics Student)  
Currently exploring AI, DL, and real-world model integration.

---