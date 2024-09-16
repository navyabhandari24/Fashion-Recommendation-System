import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

file_names = pickle.load(open('filenames.pkl', 'rb'))
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

model = ResNet50(weights='imagenet' , include_top=False , input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(feature_list.shape) # 44441 , 2048  --> 2048 features of each image

# Load and preprocess the image
img = image.load_img('sample/shopping.webp', target_size=(224, 224))
image_array = image.img_to_array(img)
expanded_image_array = np.expand_dims(image_array, axis=0)
preprocessed_image = preprocess_input(expanded_image_array)
# Extract features
result = model.predict(preprocessed_image).flatten()
# Normalize the result
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute',metric="euclidean")
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors(normalized_result.reshape(1, -1))

print(indices)

for file in indices[0]:
    temp_img = cv2.imread(file_names[file])
    cv2.imshow("output",cv2.resize(temp_img , (512,512)))
    cv2.waitKey(0)