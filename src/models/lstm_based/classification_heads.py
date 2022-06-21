import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def pretrained_classification(n_classes : int):
    x = None
    return tfkl.Dense(n_classes, activation='softmax')(x)
def dense_classification(n_classes : int):
    x = tfkl.Dense(n_classes * 2, activation='relu')
    return tfkl.Dense(n_classes, activation='softmax')(x)