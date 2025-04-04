import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


model_path = "mnist_cnn_model.h5"
if not os.path.exists(model_path):
    model = create_model()
    model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5, validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))
    model.save(model_path)
else:
    model = keras.models.load_model(model_path)

def predict_digit(img):
    img = img / 255.0 
    img = img.reshape(1, 28, 28, 1)  
    prediction = model.predict(img)
    return np.argmax(prediction)
