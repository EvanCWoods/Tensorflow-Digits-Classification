# Import libraries
import tensorflow as tf
import tensorflow.keras.datasets
from tensorflow import keras
from keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.datasets import mnist

# Create the data subsets
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


def create_model(hp):

  model = tf.keras.Sequential()

  for i in range(hp.Int('Conv layers', min_value=0, max_value=3)):
    model.add(keras.layers.Conv2D(hp.Choice(f'layers {i} filters', [16, 32, 64, 128]), 3, activation='relu'))
  
  model.add(layers.MaxPooling2D(2,2)),
  model.add(layers.Flatten()),
  model.add(layers.Dense(units=hp.Int('units',
                                      min_value=32,
                                      max_value=128,
                                      step=32),
                         activation='relu')),
  model.add(layers.Dropout(rate=0.3)),
  model.add(layers.Dense(units=hp.Int('units',
                                      min_value=32,
                                      max_value=128,
                                      step=32),
                         activation='relu')),
  model.add(layers.Dropout(rate=0.3)),
  model.add(layers.Dense(units=hp.Int('units',
                                      min_value=32,
                                      max_value=128,
                                      step=32),
                         activation='relu')),
  model.add(layers.Dropout(rate=0.3)),
  model.add(layers.Dense(10, activation='softmax')), 
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(hp.Choice('learning rate',
                                                   values=[1e-1, 1e-2, 1e-3, 1e-4])),
                metrics=['accuracy'])
  return model



tuner=RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='mnist_tuner_dir',
    project_name='mnist_test'
)

tuner.search_space_summary()


tuner.search(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels))
