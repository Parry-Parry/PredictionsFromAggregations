import os
import argparse
import logging
from pathlib import Path, PurePath
from collections import defaultdict
from src.models.lstm_based.base_model import epsilon_3_model, epsilon_5_model

from src.models.structures import *
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
parser.add_argument('--random', type=int, help='Seed for random generator')

"""
TODO:
    Once tested convert to tf functions
"""

BUFFER = 2048
BATCH_SIZE = 128
LEARNING_RATE = 0.0001

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

    name, data = retrieve_dataset(parser.dataset, parser.data_path)
    if data:
        dataset = Dataset(name, data)
    else:
        logger.error('Error in building dataset with current args')
        return 2

    logger.info('Aggregating Dataset')
    if args.partitions == 1:
        mean = 1
    else:
        mean, dataset = aggregate(dataset, args.partitions, partitions, args.seed)

    a, b, c, d = dataset.x_train.shape
    n_classes = len(np.unique(dataset.y_train))

    x_train = dataset.x_train
    y_train = tfk.utils.to_categorical(dataset.y_train, n_classes)
    x_test = dataset.x_test
    y_test = tfk.utils.to_categorical(dataset.y_test, n_classes)

    x_train, x_val, y_train, y_val = sk.model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    """I present the current worst function in the codebase"""
    tf_convert = lambda x, y, type : (tf.data.Dataset.from_tensor_slices((tf.cast(x, type), tf.cast(y, type)))).shuffle(BUFFER).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

    train_set = tf_convert(x_train, y_train, tf.uint8)
    test_set = tf_convert(x_test, y_test, tf.uint8)
    val_set = tf_convert(x_val, y_val, tf.uint8)

    logger.info('Dataset Complete')

    config = generator_config(b*c*d, 10, n_classes, 4, None, None)

    logger.info('Building Model')
    models ={
        3 : epsilon_3_model,
        5 : epsilon_5_model
    }
    try:
        model = models[args.n_gen](config)
    except KeyError:
        logger.error("No model matched n_gen value: {}".format(args.n_gen))
        return 1

    optim = tfk.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = generator_loss
    
    logger.info('Training Model for {} epochs'.format(args.stochastic, args.epochs))

    results = Result(defaultdict(list), {}, defaultdict(list))

    train_acc_metric = tfk.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tfk.metrics.SparseCategoricalAccuracy()

    for epoch in range(args.epochs):
        logger.info('Epoch {}...'.format(epoch))
        for step, (x_batch, y_batch) in enumerate(train_set): 
            with tf.GradientTape() as tape:
                pred = model(x_batch)
                loss_value = loss_fn(y_batch, pred)
            results.history[epoch].append(loss_value)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_batch, pred)

        train_acc = train_acc_metric.result()
        results.acc_score[epoch].append(train_acc)
        logger.info("Training acc over epoch: %.4f" % (float(train_acc),))

        train_acc_metric.reset_states()

        if step % BATCH_SIZE == 0:
            logger.info(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )

        for x_batch, y_batch in val_set:
            val_pred = model(x_batch, training=False)
            val_acc_metric.update_state(y_batch, val_pred)
        val_acc = val_acc_metric.result()
        results.val_acc_score[epoch] = val_acc
        val_acc_metric.reset_states()

    logger.info('Training Complete')

    test_acc_metric = tfk.metrics.SparseCategoricalAccuracy()

    for x_batch, y_batch in test_set:
        test_pred = model(x_batch, training=False)
        test_acc_metric.update_state(y_batch, test_pred)
    test_acc = test_acc_metric.result()
    results.test_acc = test_acc

    logger.info("Saving History & Models")

    """
    Save both models
    Save history + metrics
    """


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)    
    main(parser.parse_args())
