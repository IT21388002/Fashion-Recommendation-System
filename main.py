# Import necessary libraries
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Load pre-calculated features and filenames
feature_list = np.array(pickle.load(open('feature_list.pkl','rb')))
filename = pickle.load(open('filename.pkl','rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Define the Streamlit app title
st.title("TRENDZ Fashion Recommendation System")
#st.sidebar.title("What is Reco's Fashion Recommendation System")

#st.sidebar.markdown("This AI train model is based on recommend fashion items based on the features of the uploaded image.👨‍🎓👨‍🎓👨‍🎓👨‍🎓👨‍🎓")

# Function to save uploaded file
def save_uploadedfile(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function to extract features from an image
def extract_features(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

# Function to recommend fashion items based on features
def recommend(features, feature_list):
    # Change "euclidean" to "minkowski" to match the default metric in NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='auto', metric="minkowski")
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

# Handle file upload and recommendation
uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    if save_uploadedfile(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        resized_image = display_image.resize((200, 200))
        st.image(resized_image)

        # Extract features from the uploaded image
        features = extract_features(os.path.join("uploads", uploaded_file.name), model)

        # Recommend similar fashion items
        indices = recommend(features, feature_list)

        # Display recommended fashion items
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filename[indices[0][0]])
        with col2:
            st.image(filename[indices[0][1]])
        with col3:
            st.image(filename[indices[0][2]])
        with col4:
            st.image(filename[indices[0][3]])
        with col5:
            st.image(filename[indices[0][4]])
    else:
        st.error("Some error occurred in file upload...")
