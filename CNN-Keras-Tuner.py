import tensorflow as tf
import tensorflow.keras.datasets
from tensorflow import keras
from keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.datasets import mnist


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



train_data = train_data.reshape(-1,28,28,1)
test_data = test_data.reshape(-1,28,28,1)
  
# Perform the preprocessing on the training and testing data
preprocess(train_data=train_data, test_data=test_data)


print(train_data.shape)
print(test_data.shape)

def create_model(hp):

  model = tf.keras.Sequential()

  model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=train_data.shape[1:]))

  for i in range(hp.Int('Conv layers', min_value=1, max_value=5)):
    model.add(tf.keras.layers.Conv2D(hp.Choice(f'layers {i} filters', [16,32,64,128]), kernel_size=3, activation='relu'))

  model.add(tf.keras.layers.MaxPooling2D(2,2))
  model.add(tf.keras.layers.Flatten())

  for i in range(hp.Int('Dense layers', min_value=0, max_value=5)):
    model.add(tf.keras.layers.Dense(hp.Choice(f'layers {i} Units', [16, 32, 64, 128]), activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Choice(f'rate', [0.1,0.2,0.3,0.4,0.5])))

  model.add(layers.Dense(10, activation='softmax')), 
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(hp.Choice('learning rate',
                                                   values=[1e-1, 1e-2, 1e-3, 1e-4])),
                metrics=['accuracy'])
  return model


tuner=RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=1,
    directory='full_dir',
    project_name='mnist_test'
)

tuner.search_space_summary()


tuner.search(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels))
