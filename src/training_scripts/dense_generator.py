import pickle
import os
import argparse
import logging
from pathlib import Path, PurePath
from collections import defaultdict

from src.models.structures import *
from src.models.layers.custom_layers import convnet
from src.models.intermediate_robust_generator.model import *
from src.models.lstm_based.helper import retrieve_dataset, aggregate

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa

import sklearn as sk
import numpy as np

parser = argparse.ArgumentParser(description='Training of stochastic LSTM based classifier for images')

parser.add_argument('-dataset', type=str, default=None, help='Training Dataset, Supported: CIFAR10 & CIFAR100, MNIST')
parser.add_argument('-partitions', type=int, help='How much aggregation to perform upon the dataset')
parser.add_argument('-epochs', type=int, default=15, help='Number of epochs to train')
parser.add_argument('-n_gen', type=int, default=3, help='Number of generators')

parser.add_argument('--data_path', type=str, help='Training Data Path')
parser.add_argument('--partition_path', type=str, help='Where to retrieve and save aggregate data')
parser.add_argument('--dir', type=str, help='Directory to store final model')
parser.add_argument('--seed', type=int, help='Seed for random generator')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate, Default 0.0001')


BUFFER = 2048
BATCH_SIZE = 128
EPSILON = [0.001, 0.005, 0.01, 0.05, 0.1]

def main(args):
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    logger.info('Building Dataset')
    if not args.dataset and not args.data_path:
        logger.error('A dataset has not been specified')
        return 2
    
    if args.partition_path:
        """Check path formatting and either create or access partition directory"""
        p = Path(args.partition_path)
        if p.exists():
            if not p.is_dir:
                logger.warning('Invalid Partition Path, File Given')
            else:
                partitions = p
        else:
            tmp_p = PurePath(p)
            parent = tmp_p.parent

            if Path(parent.as_posix()).exists():
                p.mkdir()
            else:
                logger.warning('Invalid Directory')
    else:
        """Create or access default directory"""
        ppath = Path(os.getcwd() + 'partitions')
        if not ppath.exists(): ppath.mkdir()
        partitions = ppath

    name, data = retrieve_dataset(args.dataset, args.data_path)
    if data:
        x_train, x_test, y_train, y_test = data
        dataset = Dataset(name, x_train, x_test, y_train, y_test)
    else:
        logger.error('Error in building dataset with current args')
        return 2

    logger.info('Aggregating Dataset')
    if args.partitions == 1:
        mean = 1
    else:
        mean, shape, dataset = aggregate(dataset, args.partitions, args.partition_path, args.seed)

    if len(shape) == 3:
        a, b, c = shape
        d = 1
    else:
        a, b, c, d = shape

    n_classes = len(np.unique(dataset.y_train))

    if len(dataset.x_train.shape) == 3:
        x_train = np.expand_dims(dataset.x_train, axis=-1)
        x_test = np.expand_dims(dataset.x_test, axis=-1)
    else:
        x_train = dataset.x_train
        x_test = dataset.x_test
    y_train = tfk.utils.to_categorical(dataset.y_train, n_classes)
    y_test = tfk.utils.to_categorical(dataset.y_test, n_classes)

    """I present the current worst function in the codebase"""
    tf_convert = lambda x, y, types : (tf.data.Dataset.from_tensor_slices((tf.cast(x, types[0]), tf.cast(y, types[1])))).shuffle(BUFFER).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

    train_set = tf_convert(x_train, y_train, [tf.float32, tf.uint8])
    test_set = tf_convert(x_test, y_test, [tf.float32, tf.uint8])

    logger.info('Dataset Complete')
    intermediate = convnet
    config = generator_config((BATCH_SIZE, b, c, d), 10, n_classes, 4, intermediate, None)

    model = generator_model(config)

    optim = tfk.optimizers.Adam(learning_rate=args.lr)
    loss_fn = distance_loss

    train_acc_store = defaultdict(list)
    history = defaultdict(list)

    train_acc_metric = tfk.metrics.CategoricalAccuracy()

    for epoch in range(args.epochs):
        logger.info('Epoch {}...'.format(epoch))
        for step, (x_batch, y_batch) in enumerate(train_set): 
            with tf.GradientTape() as tape:
                pred, preds = model(x_batch, training=True)
                weights = [gen.generator.weights for gen in model.generators]
                print(weights[0])
                loss_value = loss_fn(y_batch, weights, preds)
            history[epoch].append(loss_value)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_batch, pred)

        train_acc = train_acc_metric.result()
        train_acc_store[epoch].append(train_acc)
        logger.info("Training acc over epoch: {}, loss: {}".format(float(train_acc), float(loss_value)))

        train_acc_metric.reset_states()

    logger.info('Training Complete')

    test_acc_metric = tfk.metrics.CategoricalAccuracy()

    for x_batch, y_batch in test_set:
        test_pred = model(x_batch, training=False)
        test_acc_metric.update_state(y_batch, test_pred)
    test_acc = test_acc_metric.result()

    logger.info("Test Accuracy: {}".format(test_acc))

    results = {
        'train_acc' : train_acc_store, 
        'test_acc' : test_acc, 
        'history' : history
        }

    logger.info("Saving History & Models")

    with open(args.dir + "/{}generator{}partitions{}.pkl".format(args.dataset, args.n_gen, args.partitions), 'wb') as file:
        pickle.dump(results, file)
    #model.save(args.dir + "/epsilon{}generator{}partitions{}.tf".format(epsilon, args.n_gen, args.partitions))
    
    return 0


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)    
    code = main(parser.parse_args())
    if code == 0:
        print("Executed Successfully")
    else:
        print("Error see logs")
