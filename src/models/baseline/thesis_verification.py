import sys
import os
import argparse
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

parser = argparse.ArgumentParser(description='Verification of Thesis Baseline Results')

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
        'epochs' : 15
    },
    'CIFAR' : {
        'batch_size' : 64,
        'epochs' : 30
    }
}

datasets = {
    'MNIST' : dataset_normalize(mnist.load_data(path='mnist.npz')),
    'CIFAR10' : dataset_normalize(cifar10.load_data(path='cifar10.npz')),
    'CIFAR100' : dataset_normalize(cifar100.load_data(path='cifar100.npz'))
}

mnist_shape = (28, 28, 1)
cifar_shape = (32, 32, 3) 

seed = 8008

def main():
    pass