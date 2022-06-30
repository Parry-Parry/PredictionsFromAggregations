import keras
from tensorflow.keras import layers

""" Generate Convnet

...

Parameters
-------

dim : tuple
    dimensions of input array
n_classes : int
    number of classes in dataset
"""

def gen_model(dim : tuple, n_classes : int):
    return keras.Sequential(
    [
        keras.Input(shape=dim),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",padding = 'same'),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",padding = 'same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",padding = 'same'),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",padding = 'same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ]
)