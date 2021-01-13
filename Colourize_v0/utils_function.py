import tensorflow as tf
import cv2
import numpy as np
from tf_rgb_lab_formulation import lab_to_rgb


# Loss function for model ['MSE']
def dist_loss(true_ab, predict_ab):
    return tf.math.squared_difference(true_ab, predict_ab, name='distance_loss')


# load an image and convert it to 1D gray image
def load_image(img_path):
    img = cv2.imread(img_path)
    org_rgb = cv2.resize(img, (256,256))
    gray_img = cv2.cvtColor(org_rgb, cv2.COLOR_RGB2GRAY).reshape(256,256,1)
    return org_rgb, gray_img


# processing the gray image to pass into model
def pre_process_2_colourize(gray_img):
    gray_3d = np.concatenate((gray_img, gray_img, gray_img), axis=-1) ## making the 3D image by concatenating 1D gray image
    gray_3d = gray_3d.reshape(-1, 256,256,3)
    return gray_3d/255


def predicted_rgb(gray_1d, predicted_ab):
    # Returns the predicted RGB
    # Forms predicted_Lab image by concatenating gray_1d and predicted_ab
    # Lab colorspace values in range of [L -> 0 to 100, ab -> -128 to 128] 
    predicted_Lab = np.concatenate((gray_1d* (100/255), predicted_ab*128), axis=-1)
    predicted_Lab = tf.constant(predicted_Lab.astype(np.float32))
    predicte_RGB_image = lab_to_rgb(predicted_Lab)
    return predicte_RGB_image
