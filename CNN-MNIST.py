# Import necessary libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load pre-trained CIFAR-10 model (adjust path accordingly)
@st.cache(allow_output_mutation=True)
def load_cifar_model():
    model = load_model('cifar10_cnn_model.h5')  # Ensure the model path is correct
    return model

model = load_cifar_model()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit App title
st.title("CIFAR-10 Image Classification with Pre-trained CNN")

# Display model summary
st.subheader("Model Summary")
st.text(model.summary())

# Upload image file
uploaded_file = st.file_uploader("Choose a CIFAR-10 image...", type=["jpg", "png"])

# Preprocess the uploaded image for prediction
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize to CIFAR-10 dimensions (32x32)
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict the class of the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    
    st.write(f"Predicted Class: {class_names[predicted_class]}")

# Model analysis: display accuracy and loss graphs
st.subheader("Model Performance (Training/Validation)")
if st.button("Show Accuracy and Loss"):
    # Assuming model history is saved in 'history.npy'
    history = np.load('history.npy', allow_pickle=True).item()

    # Plot model accuracy and loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax[0].plot(history['accuracy'], label='Train Accuracy')
    ax[0].plot(history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    # Plot loss
    ax[1].plot(history['loss'], label='Train Loss')
    ax[1].plot(history['val_loss'], label='Validation Loss')
    ax[1].set_title('Loss')
    ax[1].legend()

    st.pyplot(fig)
