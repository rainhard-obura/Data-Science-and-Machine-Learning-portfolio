import tensorflow as tf #type: ignore
import numpy as np
import joblib

def load_model():
    model = tf.keras.models.load_model('flood_mode.h5')
    return model

def preprocess_input(precipitation, image):
    precipitation = (precipitation - np.mean(precipitation)) / np.std(precipitation)
    #Normalize the image 
    image = image.astype(np.float32) / 255.0

    return precipitation, image