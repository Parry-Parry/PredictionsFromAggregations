import os
import argparse
import logging
from pathlib import Path, PurePath

from src.models.structures import *
from src.models.intermediate_robust_generator.model import *
from src.models.lstm_based.helper import retrieve_dataset, aggregate

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np

parser = argparse.ArgumentParser(description='Training of stochastic LSTM based classifier for images')

parser.add_argument('dataset', type=str, default=None, help='Training Dataset, Supported: CIFAR10 & CIFAR100, MNIST')
parser.add_argument('partitions', type=int, help='How much aggregation to perform upon the dataset')
parser.add_argument('epochs', type=int, default=15, help='Number of epochs to train')

parser.add_argument('--data_path', type=str, help='Training Data Path')
parser.add_argument('--partition_path', type=str, help='Where to retrieve and save aggregate data')
parser.add_argument('--dir', type=str, help='Directory to store final model')
parser.add_argument('--random', type=int, help='Seed for random generator')

"""
TODO:
    Convert Datasets to tf dataset
    Once tested convert to tf functions
"""

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

    logger.info('Dataset Complete')

    _, a, b, c = dataset.x_train.shape
    n_classes = len(np.unique(dataset.y_train))

    merger = tf.keras.layers(tf.keras.layers.LSTM(mean, activation='relu'))

    config = generator_config(a*b*c, 10, n_classes, 4, None, merger)

    logger.info('Building Model')
    model = stochastic_model(config)

    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [10000, 15000], [1e-0, 1e-1, 1e-2])
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    optim = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    loss_fn = generator_loss
    
    logger.info('Training Model for {} epochs'.format(args.stochastic, args.epochs))

    for epoch in range(args.epochs):
        logger.info('Epoch {}...'.format(epoch))
        for step, (x_batch, y_batch) in enumerate(train_dataset): # Resolve tf dataset
            with tf.GradientTape() as tape:
                pred = model(x_batch)
                loss_value = loss_fn(y_batch, pred, [gen.dense.kernel for gen in model.generators])
            grads = tape.gradient(loss_value, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))
        if step % 200 == 0:
            logger.info(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )




    logger.info('Training Complete')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)    
    main(parser.parse_args())
