import os
import argparse
import logging
from pathlib import Path, PurePath
import pickle

from src.models.lstm_based.base_model import lstm_based
from src.models.structures import *
from src.models.layers.image_layers import *
from src.models.lstm_based.classification_heads import *
from src.models.lstm_based.helper import *

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa

parser = argparse.ArgumentParser(description='Training of stochastic LSTM based classifier for images')

parser.add_argument('-dataset', type=str, default=None, help='Training Dataset, Supported: CIFAR10 & CIFAR100, MNIST')
parser.add_argument('-partitions', type=int, help='How much aggregation to perform upon the dataset')
parser.add_argument('-stochastic', default='epsilon', choices=['dense', 'reparam', 'gumbal', 'epsilon'], help='Type of Stochastic generator, defaults to naive dense')
parser.add_argument('-partition_path', default=None, help='Where to retrieve and save aggregate data')
parser.add_argument('-epochs', type=int, default=15, help='Number of epochs to train')

parser.add_argument('--data_path', type=str, help='Training Data Path')
parser.add_argument('--pretrain', help='Use a Pretrained classification head')
parser.add_argument('--dir', type=str, help='Directory to store final model')
parser.add_argument('--random', type=int, help='Seed for random generator')

generators = {
    'dense' : dense_noise_generator,
    'reparam' : dense_reparam_generator,
    'gumbal' : None,
    'epsilon' : epsilon_generator,
}

def main(args):
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    print(args.dataset)
    logger.info('Building Dataset')
    if not args.dataset and not args.data_path:
        logger.error('A dataset has not been specified')
        return 2
    
    if args.partition_path:
        """Check path formatting and either create or access partition directory"""
        p = Path(args.partition_path)
        print(p)
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
        mean, dataset = aggregate(dataset, args.partitions, partitions, args.random)

    logger.info('Dataset Complete')

    _, a, b, c = dataset.x_train.shape
    stochastic = generators[args.stochastic](a * b * c)
    lstm = Layer(mean, None)
    n_classes = len(np.unique(dataset.y_train))
    if args.pretrain:
        out = pretrained_classification(n_classes)
    else:
        out = dense_classification(n_classes)

    config = Config(stochastic, lstm, out)

    logger.info('Building Model')
    model = lstm_based(config)

    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [10000, 15000], [1e-0, 1e-1, 1e-2])
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    optim = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    loss_func = tfk.losses.CategoricalCrossentropy()
    metrics = [None]

    model.compile(optimizer=optim, loss=loss_func, metrics=metrics)
    logger.info('Training {} generation LSTM Model for {} epochs'.format(args.stochastic, args.epochs))

    history = model.fit(dataset.x_train, dataset.y_train, epochs=args.epochs)
    logger.info('Training Complete')

    with open(parser.dir, 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)    
    main(parser.parse_args())



