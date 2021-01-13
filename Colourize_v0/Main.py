import tensorflow as tf
import matplotlib.pyplot as plt
from utils_function import *
import cv2


model = tf.keras.models.load_model('models/Colourise_v0_model', custom_objects={'dist_loss': dist_loss})

org_image, gray_image = load_image('Examples/gray/Example-5.jpg')

gray_3d = pre_process_2_colourize(gray_image)

##Model gives ab values of Lab colorspace as output
predicted_ab = model.predict(gray_3d)[0]

## Converting Lab to RGB
predicted_rgb_image = predicted_rgb(gray_image, predicted_ab)

cv2.imshow('gray_image', gray_image)
plt.imshow(predicted_rgb_image)
plt.show()
