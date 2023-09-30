#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('/model.h5')

# Define a function to classify an image
def classify_image(image):
    # Preprocess the image for model input
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Predict the class (0 for real, 1 for fake)
    prediction = model.predict(image)
    return prediction[0][0]

# Create a Streamlit app
st.title("S.A.F.I.R.E. - Fake Image Detection")

# Sidebar
st.sidebar.image('path_to_sidebar_image.png', use_container_width=True)
selected_option = st.sidebar.radio("Navigation", ["Home", "Image Classification", "About the Algorithm", "FAQ"])

# Home Page
if selected_option == "Home":
    st.write("Welcome to S.A.F.I.R.E. - SYMBIOSIS ADVANCED FAKE IMAGE RECOGNITION ENGINE")
    st.write("This application can classify images as real or fake.")
    st.image('path_to_banner_image.png', use_container_width=True)

# Image Classification Page
if selected_option == "Image Classification":
    st.header("Image Classification")
    st.write("Upload an image to classify it as real or fake.")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Classify"):
            result = classify_image(image)
            if result > 0.5:
                st.write("This image is classified as FAKE.")
                st.write(f"Probability of being FAKE: {result:.2%}")
            else:
                st.write("This image is classified as REAL.")
                st.write(f"Probability of being REAL: {1 - result:.2%}")

# About the Algorithm Page
if selected_option == "About the Algorithm":
    st.header("About the Algorithm")
    st.write("S.A.F.I.R.E. uses Error Level Analysis (ELA) and Convolutional Neural Networks (CNN) to detect fake images.")
    st.write("ELA helps identify tampered regions in an image, while CNN is trained to classify images as real or fake.")
    st.write("The CNN architecture includes convolutional layers, pooling layers, dropout, and dense layers.")
    st.image('path_to_algorithm_image.png', use_container_width=True)
    st.write("For more details, please refer to the FAQ section.")

# FAQ Page
if selected_option == "FAQ":
    st.header("Frequently Asked Questions (FAQ)")
    st.write("Q1: What is Error Level Analysis (ELA)?")
    st.write("A1: Error Level Analysis is a technique used to identify areas of an image that have been modified or tampered with by analyzing compression differences.")
    
    st.write("Q2: How does S.A.F.I.R.E. work?")
    st.write("A2: S.A.F.I.R.E. uses ELA to identify potential tampered regions and a CNN to classify images as real or fake based on learned patterns.")
    
    st.write("Q3: What are the applications of this project?")
    st.write("A3: This project can be used to prevent the spread of misinformation by detecting fake or manipulated images.")
    
    # Add more FAQ items as needed
    
# Footer Image
st.image('path_to_footer_image.png', use_container_width=True)

