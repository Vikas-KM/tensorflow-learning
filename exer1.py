import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images[0].shape)
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
print(train_images.shape)

# Normalise the data - Dont forget this step else the accuracy will be very bad
train_images = train_images / 255.0
test_images = test_images / 255.0


# call back implementation
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.90:
            print('\nAccuracy of 90% reached, stopping epochs')
            self.model.stop_training = True


callbacks = MyCallback()

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

print(model.summary())

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

# callback in the fit method
model.fit(train_images, train_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)
