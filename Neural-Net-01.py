# Import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

mnist = tf.keras.datasets.mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Function to handle preprocessing the data
def preprocess(train_data, test_data):
  # Normalize the data (get all values in the tensor between 0 and 1)
  train_data = train_data / 255
  print('Train data max: ', train_data.max())
  print('Train data min: ', train_data.min())
  test_data = test_data / 255
  print('Test data max: ', test_data.max())
  print('Test data min: ', test_data.min())
  # Add a dimenstion to the tensor to represent the color channels
  train_data = tf.expand_dims(train_data, -1)
  print('Train data shape: ', train_data.shape)
  test_data = tf.expand_dims(test_data, -1)
  print('Test data shape: ',test_data.shape)
  
  # Perform the preprocessing on the training and testing data
preprocess(train_data=train_data, test_data=test_data)

# Create the first model
tf.random.set_seed(42)

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=([28,28,1])),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

history = model.fit(train_data, 
          train_labels,
          epochs=10,
          validation_data=(test_data, test_labels))

def plot_history(history):
  plt.plot(history.history['accuracy'], c='b')
  plt.plot(history.history['val_accuracy'], c='g')
  plt.title('Model accuracy (blue) vs validation accuracy (green)')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  
 
plot_history(history=history)

def predict(model, test_data):
  i = random.randint(0, 10000)
  predictions = model.predict(test_data[i])
  plt.imshow(predictions)
  plt.title(test_labels[i])

predict(model=model, test_data=test_data)
