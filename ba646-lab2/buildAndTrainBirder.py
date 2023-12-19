{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "21ec5904-7c79-4ce4-afe7-198b97533ed1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf \n",
    "from tensorflow import keras \n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b3c62e4a-8acf-4ccf-8ec7-95fd32d5195a",
   "metadata": {},
   "outputs": [],
   "source": [
    "raw_dataset=tf.data.TFRecordDataset(['src/birds-20-eachOf-358.tfrecords']) \n",
    "\n",
    "feature_description={'image':tf.io.FixedLenFeature([],tf.string), \n",
    "                     'birdType':tf.io.FixedLenFeature([],tf.int64)} \n",
    "def parse_examples(serialized_examples): \n",
    "    examples=tf.io.parse_example(serialized_examples,feature_description) \n",
    "    targets=examples.pop('birdType') \n",
    "    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg( examples['image'],channels=3),tf.float32),299,299) \n",
    "    return images,targets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8c9f8fd9-344b-42ba-bd34-ca34987cba12",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset=raw_dataset.map(parse_examples,num_parallel_calls=16).batch(16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "45677800-9b93-4560-9fa8-a5babc52df2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_size = 0\n",
    "for _ in dataset:\n",
    "    dataset_size += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a2817b99-909a-4ee8-b6c4-ce59f74464ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "raw_dataset1=tf.data.TFRecordDataset(['src/birds-10-eachOf-358.tfrecords']) \n",
    "\n",
    "feature_description={'image':tf.io.FixedLenFeature([],tf.string), \n",
    "                     'birdType':tf.io.FixedLenFeature([],tf.int64)} \n",
    "def parse_examples(serialized_examples): \n",
    "    examples=tf.io.parse_example(serialized_examples,feature_description) \n",
    "    targets=examples.pop('birdType') \n",
    "    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg( examples['image'],channels=3),tf.float32),299,299) \n",
    "    return images,targets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8ba48362-4a96-4946-a2f1-e8a28eda752a",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset1=raw_dataset1.map(parse_examples,num_parallel_calls=16).batch(16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f7effe2d-b1fd-4c0c-a99f-5d89d97bf4a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_size1 = 0\n",
    "for _ in dataset1:\n",
    "    dataset_size1 += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "be4774b6-4d9c-4b2b-827e-370b04124316",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train set size: 305\n",
      "Validation set size: 224\n",
      "Test set size: 131\n"
     ]
    }
   ],
   "source": [
    "train_size = int(0.7*dataset_size)\n",
    "valid_size = int(dataset_size1)\n",
    "test_size = dataset_size - train_size\n",
    "\n",
    "trainset = dataset.take(train_size)\n",
    "validset = dataset1.take(valid_size)\n",
    "testset = dataset.skip(train_size)\n",
    "\n",
    "# Print the number of samples in each set\n",
    "print(\"Train set size:\", train_size)\n",
    "print(\"Validation set size:\", valid_size)\n",
    "print(\"Test set size:\", test_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "0c584605-9d70-495d-ba8d-def98838e7a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocessWithAspectRatio(image,label): \n",
    "  resized_image=tf.image.resize_with_pad(image,299,299) \n",
    "  final_image=keras.applications.xception.preprocess_input(resized_image) \n",
    "  return final_image,label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6d486e16-60c8-4266-8c1c-f31292d0202d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
      "83683744/83683744 [==============================] - 0s 0us/step\n"
     ]
    }
   ],
   "source": [
    "\n",
    "trainPipe=trainset.map(preprocessWithAspectRatio,num_parallel_calls=32).cache() \n",
    "validPipe=validset.map(preprocessWithAspectRatio,num_parallel_calls=32).cache() \n",
    "testPipe=testset.map(preprocessWithAspectRatio,num_parallel_calls=32).cache() \n",
    "\n",
    "#Removing the top two layers of base model\n",
    "base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3dc6eb58-8299-45c0-98c5-7566f52fdea3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Adding new top layers\n",
    "avg=keras.layers.GlobalAveragePooling2D()(base_model.output) \n",
    "output=keras.layers.Dense(358,activation=\"softmax\")(avg) \n",
    "model=keras.models.Model(inputs=base_model.input,outputs=output) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "cfe2d481-bc31-4b17-a965-6d0c10773e16",
   "metadata": {},
   "outputs": [],
   "source": [
    "#leave bottom layers frozen\n",
    "for layer in base_model.layers: \n",
    "  layer.trainable=False "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "0e7eba33-2203-4590-ad5f-3eabebd8744b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Setting training parameters\n",
    "checkpoint_cb=keras.callbacks.ModelCheckpoint('birderModel.h5', save_best_only=True) \n",
    "earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True) \n",
    "ss=10e-1 \n",
    "optimizer=keras.optimizers.SGD(learning_rate=ss) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "f03fb564-2001-4527-821d-dc47967580e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=\"sparse_categorical_crossentropy\",optimizer=optimizer, metrics=[\"accuracy\"]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "03b1d4dc-4c86-462b-986e-ed591e276651",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/25\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-05-15 02:35:46.812945: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:428] Loaded cuDNN version 8200\n",
      "2023-05-15 02:35:54.978360: I tensorflow/compiler/xla/service/service.cc:173] XLA service 0x7f74bc01ff90 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:\n",
      "2023-05-15 02:35:54.978423: I tensorflow/compiler/xla/service/service.cc:181]   StreamExecutor device (0): Tesla T4, Compute Capability 7.5\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      1/Unknown - 13s 13s/step - loss: 5.7734 - accuracy: 0.0000e+00"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-05-15 02:35:55.686526: I tensorflow/compiler/jit/xla_compilation_cache.cc:477] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "305/305 [==============================] - 94s 264ms/step - loss: 4.0207 - accuracy: 0.1904 - val_loss: 3.0340 - val_accuracy: 0.2986\n",
      "Epoch 2/25\n",
      "305/305 [==============================] - 70s 229ms/step - loss: 1.7156 - accuracy: 0.5443 - val_loss: 2.7572 - val_accuracy: 0.3556\n",
      "Epoch 3/25\n",
      "305/305 [==============================] - 70s 231ms/step - loss: 1.0275 - accuracy: 0.7385 - val_loss: 2.6486 - val_accuracy: 0.3818\n",
      "Epoch 4/25\n",
      "305/305 [==============================] - 69s 228ms/step - loss: 0.6945 - accuracy: 0.8385 - val_loss: 2.5895 - val_accuracy: 0.3983\n",
      "Epoch 5/25\n",
      "305/305 [==============================] - 70s 230ms/step - loss: 0.5018 - accuracy: 0.9006 - val_loss: 2.5546 - val_accuracy: 0.4159\n",
      "Epoch 6/25\n",
      "305/305 [==============================] - 70s 230ms/step - loss: 0.3795 - accuracy: 0.9357 - val_loss: 2.5305 - val_accuracy: 0.4254\n",
      "Epoch 7/25\n",
      "305/305 [==============================] - 70s 231ms/step - loss: 0.2983 - accuracy: 0.9564 - val_loss: 2.5124 - val_accuracy: 0.4332\n",
      "Epoch 8/25\n",
      "305/305 [==============================] - 71s 232ms/step - loss: 0.2418 - accuracy: 0.9711 - val_loss: 2.4999 - val_accuracy: 0.4388\n",
      "Epoch 9/25\n",
      "305/305 [==============================] - 70s 229ms/step - loss: 0.2011 - accuracy: 0.9814 - val_loss: 2.4927 - val_accuracy: 0.4413\n",
      "Epoch 10/25\n",
      "305/305 [==============================] - 71s 231ms/step - loss: 0.1709 - accuracy: 0.9879 - val_loss: 2.4892 - val_accuracy: 0.4458\n",
      "Epoch 11/25\n",
      "305/305 [==============================] - 70s 228ms/step - loss: 0.1479 - accuracy: 0.9932 - val_loss: 2.4883 - val_accuracy: 0.4475\n",
      "Epoch 12/25\n",
      "305/305 [==============================] - 69s 227ms/step - loss: 0.1299 - accuracy: 0.9945 - val_loss: 2.4893 - val_accuracy: 0.4486\n",
      "Epoch 13/25\n",
      "305/305 [==============================] - 70s 229ms/step - loss: 0.1156 - accuracy: 0.9955 - val_loss: 2.4918 - val_accuracy: 0.4508\n",
      "Epoch 14/25\n",
      "305/305 [==============================] - 69s 228ms/step - loss: 0.1041 - accuracy: 0.9969 - val_loss: 2.4952 - val_accuracy: 0.4506\n",
      "Epoch 15/25\n",
      "305/305 [==============================] - 69s 226ms/step - loss: 0.0945 - accuracy: 0.9973 - val_loss: 2.4992 - val_accuracy: 0.4517\n",
      "Epoch 16/25\n",
      "305/305 [==============================] - 69s 227ms/step - loss: 0.0866 - accuracy: 0.9982 - val_loss: 2.5035 - val_accuracy: 0.4539\n",
      "Epoch 17/25\n",
      "305/305 [==============================] - 69s 228ms/step - loss: 0.0799 - accuracy: 0.9984 - val_loss: 2.5082 - val_accuracy: 0.4539\n",
      "Epoch 18/25\n",
      "305/305 [==============================] - 81s 267ms/step - loss: 0.0742 - accuracy: 0.9986 - val_loss: 2.5130 - val_accuracy: 0.4539\n",
      "Epoch 19/25\n",
      "305/305 [==============================] - 70s 229ms/step - loss: 0.0693 - accuracy: 0.9988 - val_loss: 2.5180 - val_accuracy: 0.4550\n",
      "Epoch 20/25\n",
      "305/305 [==============================] - 69s 226ms/step - loss: 0.0650 - accuracy: 0.9988 - val_loss: 2.5231 - val_accuracy: 0.4556\n",
      "Epoch 21/25\n",
      "305/305 [==============================] - 69s 228ms/step - loss: 0.0612 - accuracy: 0.9988 - val_loss: 2.5283 - val_accuracy: 0.4556\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f74c0082c10>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(trainPipe,validation_data=validPipe,epochs=25, callbacks=[checkpoint_cb,earlyStop_cb])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5b13beea-6fa3-459b-bf9f-354d6fc26789",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "436/436 [==============================] - 57s 126ms/step - loss: 198.0324 - accuracy: 0.0033 - top5: 0.1150 - top10: 0.2723 - top20: 0.4423\n"
     ]
    }
   ],
   "source": [
    "model.save('birderModel.h5')\n",
    "#load the saved model\n",
    "model=tf.keras.models.load_model('birderModel.h5') \n",
    "#how metircs we want \n",
    "top5err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,name='top5') \n",
    "top10err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10,name='top10') \n",
    "top20err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20,name='top20') \n",
    "model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer, metrics=['accuracy',top5err,top10err,top20err]) \n",
    "#returns the loss and metrics evaluated on the dataset \n",
    "resp=model.evaluate(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "bce058ec-174b-4e4a-acbd-93014ac87ef8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import zipfile\n",
    "import h5py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "dceeef44-7194-449b-a373-1c1d43a9b01b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# name of the .h5 file to be saved in the .zip file\n",
    "h5_file_name = 'birderModel.h5'\n",
    "\n",
    "# name of the .zip file to be created\n",
    "zip_file_name = 'birderModel.zip'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "57c511d8-8541-4d75-8322-2f371609c05c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a new .zip file and open it in write mode\n",
    "with zipfile.ZipFile(zip_file_name, 'w') as zip_file:\n",
    "\n",
    "    # add the .h5 file to the .zip file\n",
    "    with h5py.File(h5_file_name, 'r') as h5_file:\n",
    "        zip_file.write(h5_file_name)"
   ]
  }
 ],
 "metadata": {
  "environment": {
   "kernel": "python3",
   "name": "tf2-gpu.2-11.m108",
   "type": "gcloud",
   "uri": "gcr.io/deeplearning-platform-release/tf2-gpu.2-11:m108"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
