import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()


def preprocess(train_data, test_data):
    train_data = train_data / 255
    print('Train data min: ', train_data.min())
    print('Train data max: ', train_data.max())
    test_data = test_data / 255
    print('Test data min: ', test_data.min())
    print('test data max: ', test_data.max())

train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)

preprocess(train_data=train_data, test_data=test_data)
print(train_data.shape)
print(test_data.shape)


tf.random.set_seed(42)

class MyCallback(tf.keras.preprocessing.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.97):
            print('97% accuracy achieved, stopping training')
            self.model.stop_training = True
            
callback = myCallback()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=train_data.shape[1:]),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),

    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])    

model.compile(loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels), callbacks=[callback])
