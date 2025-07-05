import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image

# Custom Dice coefficient function
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Load the model (compile=False to avoid loss config issues)
@st.cache_resource
def load_unet_model():
    model_path = "C:/Users/Lenovo/Lung Segmentation/models/lung_segmentation_unet.h5"  # Change this to your actual model path
    return load_model(model_path, custom_objects={'dice_coef': dice_coef}, compile=False)

# Preprocess image to fit model input
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image = image.convert("L")  # Convert to grayscale for model input
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # shape: (256, 256, 1)
    img_array = np.expand_dims(img_array, axis=0)   # shape: (1, 256, 256, 1)
    return img_array


# Postprocess predicted mask to display
def postprocess_mask(mask):
    mask = mask[0, :, :, 0]  # remove batch & channel
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

# Streamlit UI
st.title("Lung Segmentation App 🫁")
st.markdown("Upload a chest X-ray or CT image to segment the lungs.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Segmenting lungs...'):
        model = load_unet_model()
        preprocessed = preprocess_image(image)
        prediction = model.predict(preprocessed)
        mask = postprocess_mask(prediction)

        # Overlay mask on original image
        overlay = np.array(image.resize((256, 256)))
        mask_3ch = np.stack([mask]*3, axis=-1)
        result = cv2.addWeighted(overlay, 0.7, mask_3ch, 0.3, 0)

        st.image(result, caption="Segmented Output", use_column_width=True)
