# 🫁 Lung Segmentation using UNet

This project focuses on **semantic segmentation** of lungs from chest X-ray images using a deep learning model based on the **UNet architecture**. It involves data preprocessing, model training, evaluation using Dice coefficient, and image mask prediction.

> 🚀 Built with TensorFlow | Medical Imaging | Segmentation

---

## 📌 Features

- ✔️ Lung mask extraction from X-ray images
- ✔️ Preprocessing pipeline (grayscale, resizing, thresholding)
- ✔️ UNet-based segmentation model
- ✔️ Custom Dice loss & Dice coefficient for training
- ✔️ Training visualization (loss vs Dice)
- ✔️ Modular project structure for expansion & deployment

---

## 🗂 Project Structure

```text
Lung-Segmentation/
├── dataset/                  # JSRT dataset (images/ + masks/)
│   ├── images/
│   └── masks/
│
├── notebooks/                # Jupyter notebooks for training
│   └── Lung_segmentation.ipynb
│
├── app/                      # Deployment code (optional)
│   └── app.ipynb
│
├── models/                   # Saved trained UNet model (.h5)
│   └── lung_segmentation_unet.h5
│
├── outputs/                  # Predicted lung masks (thresholded)
│
├── preprocessed/             # Intermediate preprocessed images
│
├── README.md                 # Project overview and usage
└── .gitignore                # Ignored files
```

## 🧠 Model: UNet for Segmentation

- Encoder-Decoder structure
- Skip connections for feature preservation
- Custom loss: **Dice Loss**
- Evaluation metrics: **Dice Coefficient**, **IoU**

---

## 🛠️ Tech Stack

- 📚 **TensorFlow/Keras**
- 🧪 **OpenCV** for image preprocessing
- 📊 **NumPy, Matplotlib**
- 🖼️ Dataset: **JSRT** (Chest X-rays with segmentation masks)

---

## ⚙️ How to Run

1. Clone the repo:

```bash
git clone https://github.com/venkatasai3005/lung-segmentation-unet.git
cd lung-segmentation-unet


cd notebooks
jupyter notebook Lung_segmentation.ipynb
```

