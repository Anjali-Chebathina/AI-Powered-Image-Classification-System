# This is app.py
# Implements:
# 2. Mini UI/Web App (Streamlit)
# 3. Loading the model for inference
# 4. Local Deployment

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2 # OpenCV

# --- Configuration ---
MODEL_PATH = 'deliverables/model_transfer_learning.h5'
INPUT_SIZE = 224

# Define the 10 class names for CIFAR-10
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# --- (TASK 3: Load Model) ---
# Use st.cache_resource for new Streamlit versions to load model
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}", icon="🚨")
        st.error(f"Please make sure the model file exists at: {MODEL_PATH}")
        st.info("You must run 'python train_model.py' first to train and save the model.", icon="ℹ️")
        return None

model = load_model()

# --- Preprocessing Function ---
def preprocess_image(img_pil):
    """
    Preprocesses a PIL image for the MobileNetV2 model.
    """
    # Convert PIL image to numpy array (OpenCV format)
    # Ensure it's RGB
    img_cv = np.array(img_pil.convert('RGB'))
    
    # Resize to the model's expected input size
    img_resized = cv2.resize(img_cv, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    
    # Expand dimensions to create a batch (1, 224, 224, 3)
    img_batch = np.expand_dims(img_resized, axis=0)
    
    # Apply MobileNetV2-specific preprocessing
    img_preprocessed = preprocess_input(img_batch.astype('float32'))
    
    return img_preprocessed

# --- (TASK 2: Mini UI/Web App) & (TASK 4: Local Deployment) ---

st.set_page_config(page_title="AI Image Classifier", layout="wide")
st.title("🖼️ AI Image Classification System (MobileNetV2)")
st.write("Upload an image to classify it into one of the 10 CIFAR-10 categories.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # 1. Display the uploaded image
        img_pil = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_pil, caption='Uploaded Image', use_column_width=True)
        
        # 2. Preprocess the image
        img_preprocessed = preprocess_image(img_pil)
        
        # 3. Make prediction
        with st.spinner('Classifying...'):
            prediction = model.predict(img_preprocessed)
        
        # 4. Display the result
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100
        
        with col2:
            st.success(f"**Prediction:** {predicted_class_name.upper()}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            # Optional: Show all probabilities
            st.write("---")
            st.write("**All Probabilities:**")
            # Create a dictionary for easier display
            probs_dict = {CLASS_NAMES[i]: f"{prediction[0][i]*100:.2f}%" for i in range(10)}
            st.json(probs_dict)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}", icon="🔥")

elif model is None:
    st.warning("Model is not loaded. Please check the terminal for errors.", icon="⚠️")