# 🧠 Brain Tumor AI (2D)

> *A deep learning project for brain tumor classification and segmentation using medical imaging. Built as a personal learning journey and portfolio showcase — not intended for clinical use.*

---

## 📌 Overview

This project explores the application of **computer vision in medical imaging**, specifically focusing on **brain tumor detection** using 2D pipelines. The goal is to build an end-to-end system that can classify and segment tumors from MRI images while also making the results more interpretable through Explainable AI (XAI).

The development began with **pure PyTorch implementations** as an exploratory phase to better understand the workflow. Later, the project transitioned to **PyTorch Lightning** to adopt a more **modular, scalable, and industry-aligned style**, making the codebase easier to maintain and extend.

Currently, the project implements a **2D pipeline**, with plans to expand toward 3D models and tabular data integration in the future.

---

## ⚙️ Pipeline Design (2D)

The system is designed in a **multi-stage pipeline**:

1. **Initial Classification**  
   - Detects whether an MRI slice contains a tumor or not.

2. **Secondary Classification**  
   - If a tumor is detected, further classifies it into **three tumor types**.  
   - Includes an additional "no tumor" check to reduce false positives.

3. **Segmentation**  
   - Localizes and segments tumor regions for a more detailed output.  
   - Currently prepared but still under fine-tuning.

This modular structure makes it easier to expand into **3D analysis** and potentially **tabular-based ML models** in future iterations.

---

To address the black-box nature of deep learning, the project integrates **XAI tools** such as Grad-CAM.  
Additionally, there are plans to extend this with **LLM-based explanations** (via API) so model predictions can be communicated in a more **human-friendly and interactive** way.

---

## 💻 Tech Stack

- **Framework:** PyTorch & PyTorch Lightning  
- **Models:** Transfer Learning & Fine-tuned CNN architectures  
- **Explainability:** Grad-CAM + LLM-based explanation (planned)  
- **Deployment:** Dashboard-ready design (future integration)  

---


## 📁 Project Structure (updated)

As part of this ongoing learning project, the codebase has been gradually transitioning to a more modular and scalable format using **PyTorch Lightning**, which better reflects modern practices and real-world applications. Earlier experiments and baseline models built using raw PyTorch have been retained under `Archives/` as a reference and learning trace.

> **Note:** The example below expands the internal structure of `Models/Classifier_Binary/`. Other models will follow a similar modular design.
```
brain-tumor-ai/
├── Archives/ # Legacy notebooks (PyTorch-based)
│
├── Models/ # Main folder for Lightning-based models
│   │
│   ├── Classifier_Binary/ # Binary classification model
│   │  │── module.py # LightningModule: defines model, loss, optimizer
│   │  ├── datamodule.py # DataModule: handles loading & preprocessing
│   │  ├── transform.py # Image augmentations (Albumentations)
│   │  ├── helper.py # Utility functions (e.g., seed, device)
│   │  ├── callbacks.py # Callbacks (checkpoint, early stopping, etc.)
│   │  ├── runner.ipynb # Notebook for training/evaluation (Colab-ready)
│   │  └── checkpoint/ # Saved model weights (.ckpt, .pt if exported)
│   │ 
│   ├── Classifier_Multiclass
│   ├── Segmentation
│
├── webapp/ # Web interface code (TBD)
├── README.md
```
---

## 🚀 Roadmap / Next Steps

- [ ] Fine-tune segmentation models for improved accuracy  
- [ ] Integrate a modern dashboard interface for visualization  
- [ ] Add LLM-powered medical explanation support  
- [ ] Extend to **3D pipelines** for volumetric data  
- [ ] Explore **tabular ML integration** for multimodal learning  

---

## ⚠️ Disclaimer

> This project is for educational purposes only.  
> The models and system are **not intended for real-world clinical use** and may not meet any regulatory standards.

---

## 👨‍💻 Author

Made by **Caynaaa** (Tech Enthusiast & Informatics Student)  
Currently exploring AI, DL, and real-world model integration.

---