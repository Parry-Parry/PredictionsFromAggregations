import numpy as np
from src.models.structures import Dataset
from tensorflow.keras.datasets import cifar100, cifar10, mnist
from sklearn.cluster import KMeans
from pathlib import Path, PurePath

def retrieve_dataset(name=None, path=None):
    normalize = lambda w, x, y, z : (w / np.float32(255), x / np.float32(255), y.astype(np.int64), z.astype(np.int64))
    if path: 
        """Not Implemented until testing complete on standard datasets"""
        (x_train, y_train), (x_test, y_test) = None
        return PurePath(path).parent.name, normalize(x_train, x_test, y_train, y_test)
    if name:
        if name == 'MNIST':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif name == 'CIFAR10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif name == 'CIFAR100':
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        else:
            return None
        return name, normalize(x_train, x_test, y_train, y_test)
    return None, None

def aggregate(data, K, dir, seed):
    pure = PurePath(dir.asposix())
    path = pure.joinpath(data.name + K + seed)

    if Path(path).exists():
        # read x data
        aggr_x = None
    else:
        x = data.x_train.flatten()
        if not seed: seed = np.random.randint(9999)
        clustering = KMeans(n_clusters=K, random_state=seed).fit_predict(x)
        aggr_x = None
    
    return Dataset(data.name, aggr_x, data.x_test, data.y_train, data.y_test)



    
    