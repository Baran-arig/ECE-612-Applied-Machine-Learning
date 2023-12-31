# -*- coding: utf-8 -*-
"""segmentImages.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ppmxqF4fWlKLHL92N8oFoeZpW3dDYXrW
"""

import tensorflow as tf 
from tensorflow import keras 
import tensorflow_datasets as tfds 
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 
import tensorflow_hub as tfhub

model=tfhub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')

import matplotlib.pyplot as plt
from PIL import Image

smsquirrels = os.listdir('squirrels')
useSquirrels = np.array(smsquirrels)[np.random.permutation(len(smsquirrels))[:9]]

# Create a directory to save the mask images
os.makedirs('squirrelMasks', exist_ok=True)

# Apply the mask code to each squirrel image
for image_file in useSquirrels:
    # Read and decode the image
    image_path = os.path.join('squirrels', image_file)
    inIm = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
    
    # Make predictions using the model
    resp = model.predict([tf.cast(inIm, tf.float32) / 255.])
    
    # Create a mask image
    mask = tf.squeeze(resp)[:, :, 18] > 0.005
    maskIm = np.zeros_like(inIm)  # Initialize mask with white pixels
    
    if np.any(mask):
        maskIm[mask] = [255,255,255]  # Set masked pixels to black
    
    # Save the mask image as JPEG
    mask_file = os.path.join('squirrelMasks', f'mask_{image_file[:-4]}.jpg')
    mask_image = Image.fromarray(maskIm)
    mask_image.save(mask_file, format='JPEG')
    
    # Display the original and masked images
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(maskIm)
    plt.title('Masked Image')
    plt.subplot(1, 2, 2)
    plt.imshow(inIm)
    plt.title('Original Image')
    plt.show()

smsquirrels = os.listdir('birds/')
useSquirrels = np.array(smsquirrels)[np.random.permutation(len(smsquirrels))[:5]]

# Create a directory to save the mask images
os.makedirs('birdMasks', exist_ok=True)

# Apply the mask code to each squirrel image
for image_file in useSquirrels:
    # Read and decode the image
    image_path = os.path.join('birds', image_file)
    inIm = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
    
    # Make predictions using the model
    resp = model.predict([tf.cast(inIm, tf.float32) / 255.])
    
    # Create a mask image
    mask = tf.squeeze(resp)[:, :, 17] > 0.4
    maskIm = np.zeros_like(inIm)  # Initialize mask with white pixels
    
    if np.any(mask):
        maskIm[mask] = [255,255,255]  # Set masked pixels to black
    
    # Save the mask image as JPEG
    mask_file = os.path.join('birdMasks', f'mask_{image_file[:-4]}.jpg')
    mask_image = Image.fromarray(maskIm)
    mask_image.save(mask_file, format='JPEG')
    
    # Display the original and masked images
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(maskIm)
    plt.title('Masked Image')
    plt.subplot(1, 2, 2)
    plt.imshow(inIm)
    plt.title('Original Image')
    plt.show()