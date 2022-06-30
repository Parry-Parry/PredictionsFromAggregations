import sys
import os
import pathlib
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from tensorflow.keras.datasets import cifar100, cifar10, mnist

from src.models.thesis_baseline.helper import *

cwd = Path(os.getcwd())
root = cwd.parent.parent.parent
data = pathlib.PurePath(root, 'data')
interim = pathlib.PurePath(data, 'interim')

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

datasets = {
    'MNIST' : {
        'shape' : (28, 28, 1)
        },
    'CIFAR10' : {
        'shape' : (32, 32, 3) 
    },
    'CIFAR100' : {
        'shape' : (32, 32, 3) 
    }
}

seed = 8008

def partitions(dataset):
    
    x_train = v['data']['x_train']

    for k in tqdm(test_param_grid[1]['K']):
        print("Generating Partitions for {} dataset with {} clusters".format(key, k))
        x_vecs = flatten(x_train)
        x, y = partition(x_vecs, k, SEED=seed, write_path=pathlib.PurePath(interim, key + str(k) + '_partitions.tsv'))

def gaussianParams():
    storage = {}

    for k in tqdm(test_param_grid[1]['K']):
        dir = pathlib.PurePath(partition_dir, key + str(k) + '_partitions.tsv')
        if Path(dir.as_posix()).exists():
            print("Loading Partitions for {} dataset with {} clusters".format(key, k))
            with open(dir) as f:
                lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            x = []
            y = []
            for line in lines:
                tokens = line.split()
                x_vec = np.zeros(len(tokens)-1)
                for i in range(len(tokens)-1):
                    x_vec[i] = float(tokens[i])

                x.append(x_vec)
                y.append(int(tokens[-1]))

        print("Generating Mean & Variance for {} dataset with {} clusters".format(key, k))
        members = groupClusters(x, y)
        mu, sigma = computeMultivariateGaussianParameters(members)

        storage[k] = (mu, sigma)

        # Explicit Garbage Collection to Save Memory
        
        del x 
        del y
        del members
        del mu
        del sigma

    print("Saving Mean & Variance for {} dataset".format(key))
    with open(pathlib.PurePath(interim, "meanvar{}.pkl".format(key)), 'wb') as f:
        pickle.dump(storage, f)

if __name__ == "__main__":
    main()