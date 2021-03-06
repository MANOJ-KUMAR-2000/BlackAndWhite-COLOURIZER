import tensorflow as tf
import matplotlib.pyplot as plt
from utils_function import *
import cv2

model = tf.keras.models.load_model('models/Colourize_v1_reuidalModel')

org_image, gray_image = load_image_from_path('Examples/gray.jpg')

gray_3d = pre_process_2_colourize(gray_image)

predicted_ab = model.predict(gray_3d)[0]

predicted_rgb_image = predicted_rgb(gray_image, predicted_ab)

cv2.imshow('gray_image', gray_image)
plt.imshow(predicted_rgb_image)
plt.show()
