import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('densenet121_finetuned.h5')

# Class labels in the order your model outputs predictions
class_labels = ['Glioma', 'Pituitary', 'Meningioma', 'No Tumor']

st.title("Brain Tumor Classification Model")
st.write("Upload an MRI image to detect brain tumors or choose a sample image.")

# Sample images with captions
sample_images = {
    "Glioma Tumor": ["sample/glioma.jpg", "sample/glioma1.jpg", "sample/glioma2.jpg"],
    "Pituitary Tumor": ["sample/pitituary.jpg", "sample/pitituary1.jpg", "sample/pitituary2.jpg"],
    "Meningioma Tumor": ["sample/meninsia.jpg", "sample/meningioma1.jpg", "sample/meningioma2.jpg"],
    "No Tumor": ["sample/notumor.jpg", "sample/notumor1.jpg", "sample/notumor2.jpg"]
}

st.sidebar.header("Sample Images")
st.sidebar.write("All these sample images are cleaned and processed images.")
selected_category = st.sidebar.selectbox("Choose a tumor category to test:", list(sample_images.keys()))
selected_sample = st.sidebar.selectbox("Choose an image:", sample_images[selected_category])

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Load selected sample if no uploaded image is provided
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', clamp=True, output_format='JPEG')
else:
    image = Image.open(selected_sample).convert('RGB')
    st.image(image, caption=f'Sample Image: {selected_category}', clamp=True, output_format='JPEG')

# Preprocess the image
img_array = np.array(image.resize((224, 224))) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make predictions
prediction = model.predict(img_array)

# Get the class with the highest probability
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

st.write(f"Prediction: **{class_labels[predicted_class]}**")
st.write(f"Confidence: {confidence:.2f}")

# Debug output (optional)
# st.write(f"Raw prediction: {prediction}")
