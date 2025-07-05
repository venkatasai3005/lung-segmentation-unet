import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

# Load the UNet model
@st.cache_resource
def load_unet_model():
    return load_model("lung_segmentation_unet.h5", compile=False)

# Preprocess uploaded image
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((256, 256))                # Resize to model input size
    img_array = np.array(image) / 255.0             # Normalize
    return img_array.reshape(1, 256, 256, 1)         # Add batch and channel dimensions

# Load model
model = load_unet_model()

# Streamlit UI
st.title("🫁 Lung Segmentation App")
st.write("Upload a chest X-ray image to get the segmented lung mask using UNet.")

uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess and predict
    preprocessed = preprocess_image(uploaded_file)
    prediction = model.predict(preprocessed)[0, :, :, 0]  # Remove batch and channel dims

    # Postprocess mask (optional thresholding)
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255

    # Show prediction
    st.image(binary_mask, caption="Predicted Lung Mask", use_column_width=True, clamp=True)
