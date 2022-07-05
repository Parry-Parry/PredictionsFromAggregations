import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def pretrained_classification(n_classes : int, in_dim : tuple):
    x = tfk.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights='imagenet', input_shape=in_dim)
    x.trainable = False
    x = tfkl.Dense(n_classes * 2, activation='relu')(x)
    return tfkl.Dense(n_classes, activation='softmax')(x)
def dense_classification(n_classes : int):
    x = tfkl.Dense(n_classes * 2, activation='relu')
    return tfkl.Dense(n_classes, activation='softmax')(x)