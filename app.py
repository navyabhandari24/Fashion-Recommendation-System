import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm # tell the progress of for loop
import pickle

# inculde_top - we remove the top layer and create a new one
# input_shape - size of the image which is given to the resnet model - (224,224,3) standard size of the image
model = ResNet50(weights='imagenet' , include_top=False , input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(expanded_image_array)
    # Extract features
    result = model.predict(preprocessed_image).flatten()
    # Normalize the result
    normalized_result = result / norm(result)
    return normalized_result


file_names = []
for file in os.listdir('images'):
    file_names.append(os.path.join("images", file))

feature_list = [] # 2d list
for file in tqdm(file_names):
     feature_list.append(extract_features(file,model))


pickle.dump(feature_list , open('embeddings.pkl', 'wb'))
pickle.dump(file_names , open('filenames.pkl', 'wb'))
