from src.models.lstm_based.base_model import lstm_based
from src.models.structures import *
from src.models.layers.dense import dense_generator
from src.models.layers.reparam import dense_reparam_generator
from src.models.layers.gumbal import * # Change when complete
from src.models.layers.epsilon import epsilon_generator
from src.models.structures import *

import argparse
import logging

parser = argparse.ArgumentParser(description='Training of stochastic LSTM based classifier')

parser.add_argument('data_path', type=str, help='Training Data Path, expects Tensorflow dataset in directory')
parser.add_argument('stochastic', type=str, default='dense', help='Type of Stochastic generator, defaults to naive dense')
parser.add_argument('epochs', type=int, default=15, help='Number of epochs to train')

parser.add_argument('--resnet', help='Use a Pretrained Resnet classification head')
parser.add_argument('--dir', type=str, help='Directory to store final model')


def retrieve_dataset(path : str):
    pass


def main(args):
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    logger.info('Building Dataset')
    data = Dataset(retrieve_dataset(parser.data_path))


    config = None

    logger.info('Building Model')
    model = lstm_based(config)

    optim=None
    loss_func = None
    metrics = [None]

    model.compile(optimizer=optim, loss=loss_func, metrics=metrics)
    logger.info('Training {} generation LSTM Model for {} epochs'.format(args.stochastic, args.epochs))

    history = model.fit(data.x_train, data.y_train, epochs=args.epochs)
    logger.info('Training Complete')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)    
    main(parser.parse_args())




