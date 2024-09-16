import sys
import tensorflow
import streamlit as st
import os
from PIL import Image
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import numpy as np
from sklearn.neighbors import NearestNeighbors

file_names = pickle.load(open('filenames.pkl', 'rb'))
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

st.title("Fashion Recommender System")


def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join("upload", uploaded_image.name), "wb") as f:
            f.write(uploaded_image.getbuffer())
        return 1
    except:
        return 0


uploaded_file = st.file_uploader("Choose an image")

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric="euclidean")
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices


def feature_extraction(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(expanded_image_array)
    # Extract features
    result = model.predict(preprocessed_image).flatten()
    # Normalize the result
    normalized_result = result / norm(result)
    return normalized_result


if uploaded_file is not None:
    if save_uploaded_image(uploaded_file):
        # Display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image, width=200)
        features = feature_extraction(os.path.join("upload", uploaded_file.name), model)

        indices = recommend(features, feature_list)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(file_names[indices[0][0]], width=120)

        with col2:
            st.image(file_names[indices[0][1]], width=120)

        with col3:
            st.image(file_names[indices[0][2]], width=120)

        with col4:
            st.image(file_names[indices[0][3]], width=120)

        with col5:
            st.image(file_names[indices[0][4]], width=120)
    else:
        st.header("Error occurred while uploading the image")
