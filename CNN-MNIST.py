import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained CNN model
@st.cache(allow_output_mutation=True)
def load_cnn_model():
    model = load_model('mnist_cnn_model.h5')  # Ensure to have this pre-trained model
    return model

model = load_cnn_model()

# Title of the Streamlit app
st.title("MNIST Handwritten Digit Classification")

# Upload an image
uploaded_file = st.file_uploader("Upload a handwritten digit image (28x28 pixels, grayscale)", type=["png", "jpg", "jpeg"])

# Preprocess the image for the CNN model
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
    return image

# Display the uploaded image and make predictions
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    st.write(f"Predicted Digit: {predicted_class}")

# Display model analysis (accuracy and loss graphs)
st.subheader("Model Performance Metrics")

# Load model history (training history)
@st.cache
def load_history():
    history = np.load('history.npy', allow_pickle=True).item()
    return history

history = load_history()

# Plot accuracy and loss graphs
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax[0].plot(history['accuracy'], label='Training Accuracy')
ax[0].plot(history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Model Accuracy')
ax[0].legend()

# Loss plot
ax[1].plot(history['loss'], label='Training Loss')
ax[1].plot(history['val_loss'], label='Validation Loss')
ax[1].set_title('Model Loss')
ax[1].legend()

st.pyplot(fig)


