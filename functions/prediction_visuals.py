import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras


def prediction_visuals(image,model, img_height, img_width, class_names):            
    img = tf.keras.utils.load_img(
        image, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    '\033[1m' +  "This image most likely belongs to {} ."
    .format(class_names[np.argmax(score)]))
    return PIL.Image.open(image)