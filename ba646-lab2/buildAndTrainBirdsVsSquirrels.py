# -*- coding: utf-8 -*-
"""buildAndTrainBirdsVsSquirrels.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SXZCUcihtKCVmdWctLmRfeQ1EEZyf_Rv
"""

import tensorflow as tf 
from tensorflow import keras 
#import tensorflow_datasets as tfds 
import numpy as np

raw_dataset=tf.data.TFRecordDataset(['/content/drive/MyDrive/Applied ML/Assignment 2/birds-vs-squirrels-train.tfrecords']) 
feature_description={'image':tf.io.FixedLenFeature([],tf.string), 
                     'label':tf.io.FixedLenFeature([],tf.int64)} 
def parse_examples(serialized_examples): 
  examples=tf.io.parse_example(serialized_examples,feature_description) 
  targets=examples.pop('label') 
  images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg( 
      examples['image'],channels=3),tf.float32),299,299) 
  return images,targets

dataset=raw_dataset.map(parse_examples,num_parallel_calls=16).batch(16)

dataset_size = 0
for _ in dataset:
    dataset_size += 1

# Split the dataset into train and test sets
train_size = int(0.6 * dataset_size)
valid_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - valid_size

trainset = dataset.take(train_size)
validset = dataset.skip(train_size).take(valid_size)
testset = dataset.skip(train_size + valid_size)

print("Train set size:", train_size)
print("Validation set size:", valid_size)
print("Test set size:", test_size)

def preprocessWithAspectRatio(image,label): 
  resized_image=tf.image.resize_with_pad(image,299,299) 
  final_image=keras.applications.xception.preprocess_input(resized_image) 
  return final_image,label

trainPipe=trainset.map(preprocessWithAspectRatio,num_parallel_calls=32).cache() 
validPipe=validset.map(preprocessWithAspectRatio,num_parallel_calls=32).cache() 
testPipe=testset.map(preprocessWithAspectRatio,num_parallel_calls=32).cache()

base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)

avg=keras.layers.GlobalAveragePooling2D()(base_model.output) 
output=keras.layers.Dense(3,activation="softmax")(avg) 
model=keras.models.Model(inputs=base_model.input,outputs=output) 
model.summary()

for layer in base_model.layers: 
  layer.trainable=False

checkpoint_cb=keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True) 
earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True) 
ss=5e-1

optimizer=keras.optimizers.SGD(learning_rate=ss)

model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"]) 
model.fit(trainPipe,validation_data=validPipe,epochs=25, callbacks=[checkpoint_cb,earlyStop_cb])

import zipfile
import h5py

h5_file_name = 'model.h5'

zip_file_name = 'birdsVsSquirrelsModel.zip'

with zipfile.ZipFile(zip_file_name, 'w') as zip_file:

    # add the .h5 file to the .zip file
    with h5py.File(h5_file_name, 'r') as h5_file:
        zip_file.write(h5_file_name)