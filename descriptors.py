import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Charger le modèle une seule fois
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def vgg16_descriptor(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = vgg_model.predict(img_array)
    return features.flatten()





def color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    return hist.flatten()

def gray_histogram(image_path, bins=256):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    return hist.flatten()

def correlogram(image_path, distance=5):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    glcm = graycomatrix(image, distances=[distance], angles=[0], levels=256, symmetric=True, normed=True)
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return correlation
