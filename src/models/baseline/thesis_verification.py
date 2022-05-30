import sys
import os
import logging
import pathlib
from pathlib import Path

import sklearn
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100, cifar10, mnist

from helper import *

cwd = Path(os.getcwd())
root = cwd.parent.parent
data = pathlib.PurePath(root, 'data')
history = pathlib.PurePath(data, 'history')
results = pathlib.PurePath(data, 'results', 'thesis')

test_param_grid ={
    1 : {
        'K' : [50, 100, 200, 500, 1000],
        'epsilon' : 0.01
    },
    2 : {
        'K' : 1000,
        'epsilon' : [0.01, 0.05, 0.1, 1.0]
    }
}

model_param_grid = {
    'MNIST' : {
        'batch_size' : 1000,
        'epochs' : 15,
        'save_history' : False,
        'path' : history
    },
    'CIFAR10' : {
        'batch_size' : 64,
        'epochs' : 30,
        'save_history' : False,
        'path' : history
    },
    'CIFAR100' : {
        'batch_size' : 64,
        'epochs' : 30,
        'save_history' : False,
        'path' : history
    }
}

datasets = {
    'MNIST' : {
        'data' : dataset_normalize(mnist.load_data(path='mnist.npz')),
        'shape' : (28, 28, 1)
        },
    'CIFAR10' : {
        'data' : dataset_normalize(cifar100.load_data(path='cifar10.npz')),
        'shape' : (32, 32, 3) 
    },
    'CIFAR100' : {
        'data' : dataset_normalize(cifar100.load_data(path='cifar100.npz')),
        'shape' : (32, 32, 3) 
    }
}

mnist_shape = (28, 28, 1)
cifar_shape = (32, 32, 3) 

seed = 8008

def main():
    results1 = pd.DataFrame()
    results2 = pd.DataFrame()

    for key, v in datasets.items():
        x_train = v['data']['x_train']
        x_test = v['data']['x_test']
        y_train = v['data']['y_train']
        y_test = v['data']['y_test']

        

        logging.info("Running Test 1...")

        for k in test_param_grid[1]['K']:
            results = runTest(k, test_param_grid[1]['epsilon'], (x_train, x_test), (y_train, y_test), v['shape'], model_param_grid[key], partition_out=pathlib.PurePath(data, 'interim', str(k) + key + '_partitions.tsv'))
            results['dataset'] = key
            results['K'] = k

            results1.append(results)

        logging.info("Test 1 Completed Successfully")

        logging.info("Running Test 2...")

        for e in test_param_grid[2]['epsilon']:
            results = runTest(test_param_grid[2]['K'], e, (x_train, x_test), (y_train, y_test), v['shape'], model_param_grid[key], partition_out=pathlib.PurePath(data, 'interim', str(k) + key + '_partitions.tsv'))
            results['dataset'] = key
            results['Epsilon'] = e

            results2.append(results)

        logging.info("Test 2 Completed Successfully")

        logging.info("Saving Results...")

        results1.to_csv(pathlib.PurePath(results, 'test1.csv'))
        results2.to_csv(pathlib.PurePath(results, 'test2.csv'))



