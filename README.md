# 🦴 Bone Fracture AI

A deep learning project for classifying X-Ray images into two categories:

- **Fractured**
- **Not Fractured**

The project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and provides a web interface for uploading X-Ray images and getting a prediction.

---

## 📌 Project Overview

This project demonstrates how a trained CNN model can be integrated into a complete application.

The system consists of three main parts:

- **CNN Model** — TensorFlow/Keras model used for image classification.
- **FastAPI Backend** — Receives the uploaded image and performs the prediction.
- **Streamlit Frontend** — Provides the user interface for uploading and analyzing X-Ray images.

---

## 🏗️ Project Structure

```text
X_Ray_Classification/
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── model.py
│       └── schema.py
│
├── frontend/
│   └── app.py
│
├── model/
│   └── bone_fracture_model.keras
│
├── .gitignore
└── README.md
