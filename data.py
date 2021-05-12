# Import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Import the mnist dataset
mnist = tf.keras.datasets.mnist

# Create the datasets
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Function to show information on the dataset (find what preprocessing is)
def train_test_features(train_data, test_data):
  print('Train data max: ', train_data.max())
  print('Train data min: ', train_data.min())
  print('Test data max: ', test_data.max())
  print('Test data min: ', test_data.min())
  print('Train data shape: ', train_data.shape)
  print('Test data shape: ',test_data.shape)
  
# Call the function to find informatin on the data
train_test_features(train_data=train_data, test_data=test_data)

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

# Function to plot example images in the range 0-1000 at random
def plot_raw_images(data):
  i = random.randint(0,1000)
  plt.imshow(data[i])
  
# Show sample images from the training datset
plot_raw_images(data=train_data)
