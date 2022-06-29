import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def create_dense_classifier(input : tuple, n_classes):
    return tfk.Sequential(
        [
            tfkl.Dense(512, input_shape=input, activation=tf.nn.leakyRelu),
            tfkl.Dense(256, activation=tf.nn.leakyRelu),
            tfkl.Dense(128, activation=tf.nn.leakyRelu),
            tfkl.Dense(n_classes, activation='softmax'),
        ]
    )

def gen_model(input : tuple, n_classes : int):
    return tfk.Sequential(
    [
        tfk.Input(shape=input),
        tfkl.Conv2D(32, kernel_size=(3, 3), activation="relu",padding='same'),
        tfkl.Conv2D(32, kernel_size=(3, 3), activation="relu",padding='same'),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Dropout(0.25),
        tfkl.Conv2D(64, kernel_size=(3, 3), activation="relu",padding='same'),
        tfkl.Conv2D(64, kernel_size=(3, 3), activation="relu",padding='same'),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Dropout(0.25),
        tfkl.Flatten(),
        tfkl.Dense(512, activation='relu'),
        tfkl.Dropout(0.5),
        tfkl.Dense(n_classes, activation="softmax"),
    ]
)

def create_efficientnet_classifier(input : tuple, n_classes):
    net = tfk.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights='imagenet', input_shape=input)
    net.trainable = False
    return tfk.Sequential(
        [
            net,
            tfkl.Dense(256, activation='relu'),
            tfkl.Dense(n_classes, activation='softmax')
        ]
    )