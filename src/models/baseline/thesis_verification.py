import sys
import os
import logging
import argparse
import pathlib
from pathlib import Path

import sklearn
import pandas as pd
import numpy as np
from tqdm import tqdm

from tensorflow import keras
from src.models.baseline.kmeans import runKmeans
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100, cifar10, mnist

from helper import *

### CMD ARGS ###

parser = argparse.ArgumentParser("Verification of Baseline Results")
parser.add_argument("--partition_dir", help="Where cached partitions are found")
parser.add_argument("--seed", help="Random State Seed", type="int")

cwd = Path(os.getcwd())
root = cwd.parent.parent
data = pathlib.PurePath(root, 'data')
history = pathlib.PurePath(data, 'history')
interim = pathlib.PurePath(data, 'interim')
results = pathlib.PurePath(data, 'results', 'thesis')

### PARAMETERS ###

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

### DATASET INITIALIZATION ###

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

seed = 8008

def main():
    args = parser.parse_args()

    results1 = pd.DataFrame()
    results2 = pd.DataFrame()
    
    for key, v in datasets.items():
        x_train = v['data']['x_train']
        x_test = v['data']['x_test']
        y_train = v['data']['y_train']
        y_test = v['data']['y_test']

        logging.info("Running Test 1 on {}...".format(key))

        for k in tqdm(test_param_grid[1]['K']):
            if args.partition_dir:
                dir = args.partition_dir + "/" + key + k + '_partitions.tsv'
                logging.info("Loading Partitions for {} dataset with {} clusters".format(key, k))
                with open(dir) as f:
                    lines = f.readlines()
                    lines = [line.rstrip() for line in lines]
                    for line in lines:
                        tokens = line.split()
                        x_vec = np.zeros(len(tokens)-1)
                        for i in range(len(tokens)-1):
                            x_vec[i] = float(tokens[i])

                        x.append(x_vec)
                        y.append(int(tokens[-1]))

            else:
                logging.info("Generating Partitions for {} dataset with {} clusters".format(key, k))
                x_vecs = flatten(x_train)
                x, y = partition(x_vecs, k, SEED=seed, path="", write_path=pathlib.PurePath(interim, key + k + '_partitions.tsv'))

            kmeans = runKmeans(k,  (x_train, x_test), (y_train, y_test), v['shape'], model_param_grid[key])
            gauss, epsilon, complete = runTest(k, test_param_grid[1]['epsilon'], (x_train, x_test), (y_train, y_test), (x, y), v['shape'], model_param_grid[key], partition_out=pathlib.PurePath(data, 'interim', str(k) + key + '_partitions.tsv'))
            
            sets = [kmeans, gauss, epsilon, complete]
            for set in sets:
                set['dataset'] = key
                set['K'] = k
                results1.append(set)

        logging.info("Test 1 Completed Successfully")

        logging.info("Running Test 2 on {}...".format(key))

        for e in tqdm(test_param_grid[2]['epsilon']): # TODO: Make function run test 1 or test 2
            _, results, _ = runTest(test_param_grid[2]['K'], e, (x_train, x_test), (y_train, y_test), (x, y), v['shape'], model_param_grid[key], partition_out=pathlib.PurePath(data, 'interim', str(k) + key + '_partitions.tsv'))
            results['dataset'] = key
            results['Epsilon'] = e

            results2.append(results)

        logging.info("Test 2 Completed Successfully")

        logging.info("Saving Results...")

        results1.to_csv(pathlib.PurePath(results, 'test1.csv'))
        results2.to_csv(pathlib.PurePath(results, 'test2.csv'))



