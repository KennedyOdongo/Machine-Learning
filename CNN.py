#!/usr/bin/env python
# coding: utf-8

#  ## Convolutional Neural Networks

# ### First Toy NN model using Tensor Flow,Keras and Fashion MNIST dataset.

# In[1]:


#import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, Nadam
from keras.callbacks import TensorBoard
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#load the data set from the API
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


#Take a look at the shape of the data set
print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)


# In[4]:


#reshape the images to have one more dimension which is the color=1, grayscale as in the problem
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))


# In[5]:


#normalize the data set, the pixels range from 0 to 255
train_images_norm = train_images / 255.0
test_images_norm = test_images / 255.0


# #### This network architecture contains and CNN layers, followed by 1 pooling layer and finally a connected layer, All activation function are ReLU

# In[6]:


model=tf.keras. Sequential([
layers.Conv2D(8,5,padding="same",strides=[2,2], activation='relu', input_shape=(28,28,1) ),
layers.Conv2D(16,3,padding="same",strides=[2,2], activation='relu'),
layers.Conv2D(32,3,padding="same",strides=[2,2], activation='relu'),
layers.Conv2D(32,3,padding="same",strides=[2,2], activation='relu'),
layers.AveragePooling2D(),
layers.Flatten(),
layers.Dense(128,activation="relu"),
layers.Dense(10),
layers.Softmax()])


# In[7]:


model.summary()


# In[8]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


history=model.fit(train_images_norm, train_labels, epochs=10, batch_size=1028, shuffle=True, validation_split=0.1)


# In[10]:


test_loss, test_accuracy=model.evaluate(test_images,test_labels)
print("The testing accuracy is {}".format(test_accuracy))


# In[12]:


plt.plot(history.history["acc"],label="training accuracy")
plt.plot(history.history["val_acc"],label="validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation accuracy")
plt.legend(['train', 'val'],loc='lower right')
plt.ylim([0.5,1])


# ## Training the NN with Batch Normalization

# ##### Batch normalization, or batchnorm for short, is proposed as a technique to help coordinate the update of multiple layers in the model.

# In[13]:


model_1 = tf.keras.Sequential([
  layers.Conv2D(8, 5, padding = 'same', strides = 2, input_shape = (28, 28, 1), activation = 'relu'),
  layers.BatchNormalization(),


  layers.Conv2D(16, 3, padding = 'valid', strides = 2, activation = 'relu'),
  layers.BatchNormalization(),
  layers.AveragePooling2D(1),
  layers.Dropout(0.25),


  layers.Conv2D(32, 3, padding = 'valid', strides = 2, activation = 'relu'),
  layers.BatchNormalization(),
  layers.Dropout(0.25),


  layers.Conv2D(32, 2, padding = 'valid', strides = 2, activation = 'relu'),
  layers.BatchNormalization(),
  layers.AveragePooling2D(1),
  layers.Dropout(0.25),

  layers.Flatten(),

  layers.Dense(512, activation='relu'),
  layers.BatchNormalization(),
  layers.Dropout(0.5),

  layers.Dense(128, activation='relu'), 
  layers.BatchNormalization(),
  layers.Dropout(0.5),

  layers.Dense(10, activation='softmax')])


# In[14]:


model_1.summary()


# In[15]:


model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history1=model_1.fit(train_images_norm, train_labels, epochs=10, batch_size=1028, shuffle=True, validation_split=0.1)


score_1 = model_1.evaluate(test_images_norm, test_labels, verbose=1)


# In[17]:


history1.history.keys()


# #### Visualizing results.

# In[18]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy metrics for model 1 ')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

