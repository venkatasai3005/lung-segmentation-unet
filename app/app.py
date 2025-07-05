import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from PIL import Image

# Custom Dice Coefficient Metric for loading model
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Load model
@st.cache_resource
def load_unet_model():
    model_path = "models/lung_segmentation_unet.h5"
    if not os.path.exists(model_path):
        st.error("Model file not found at 'models/lung_segmentation_unet.h5'")
        return None
    return load_model(model_path, custom_objects={'dice_coef': dice_coef}, safe_mode=True)

model = load_unet_model()

# Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image.convert("L")) / 255.0
    image = np.expand_dims(image, axis=-1)  # (256, 256, 1)
    image = np.expand_dims(image, axis=0)   # (1, 256, 256, 1)
    return image

# Postprocessing function
def postprocess_mask(mask):
    mask = (mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255  # Thresholding
    return mask

# UI
st.set_page_config(page_title="Lung Segmentation App", layout="centered")
st.title("Lung Segmentation using UNet")
st.write("Upload a chest X-ray image to get lung segmentation output.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model is not None:
        input_img = preprocess_image(image)
        pred_mask = model.predict(input_img)
        result_mask = postprocess_mask(pred_mask)

        st.subheader("Predicted Lung Mask")
        st.image(result_mask, clamp=True, use_column_width=True)
    else:
        st.warning("Model could not be loaded.")
