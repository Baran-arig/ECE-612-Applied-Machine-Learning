# -*- coding: utf-8 -*-
"""generateBirdFeederImagesFromText.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16eJ9xKCRB0iumkI5HbYtiVKAG6DB6KH_
"""

import matplotlib.pyplot as plt
import numpy as np
from keras_cv.models import StableDiffusion

model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)

prompts = [
    "Squirrels and birds at bird feeder",
    "birds and squirrels at bird feeder",
    "birds and squirrels competing for food at a bird feeder",
    "squirrels and birds fighting over food at a bird feeder",
    "food in bird feeder getting fought over by birds and squirrels"
]

generated_images = []
for prompt in prompts:
    img = model.text_to_image(
        prompt=prompt,
        batch_size=1,  # How many images to generate at once
        num_steps=25,  # Number of iterations (controls image quality)
        seed=123  # Set this to always get the same image from the same prompt
    )
    generated_images.append(img)

for i, img in enumerate(generated_images):
    plt.imshow(np.squeeze(img))
    plt.savefig(f"fromText_{i+1}.jpg", format='jpg')
    plt.show()