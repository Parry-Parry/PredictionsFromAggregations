import collections
from pathlib import Path, PurePath
import pickle 

from src.models.structures import Dataset

import numpy as np
from tensorflow.keras.datasets import cifar100, cifar10, mnist
from sklearn.cluster import KMeans


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
    pure = PurePath(dir)
    path = pure.joinpath(data.name + str(K) + str(seed) + '.pkl')

    if Path(path).exists():
        with open(path, 'rb') as f:
            tmp = pickle.load(f)
            aggr_x, aggr_y, avg = tmp
            shape = aggr_x.shape
    else:
        shape = tuple([K] + list(data.x_train.shape[1:]))
        x = np.array([img.flatten() for img in data.x_train])
        
        if not seed: seed = np.random.randint(9999)
        clustering = KMeans(n_clusters=K, random_state=seed).fit_predict(x)

        cluster_members =  collections.defaultdict(list)
        cluster_labels = collections.defaultdict(list)
        for a, b, c in zip(x, data.y_train, clustering): 
            cluster_members[c].append(a)
            if 'CIFAR' in data.name:
                cluster_labels[c].append(b[0])
            else:
                cluster_labels[c].append(b)
        
        centroids = []
        labels = []
        member_count = []

        for k, v in cluster_members.items():
            centroids.append(np.mean(v, axis=0))
            vals, counts = np.unique(cluster_labels[k], return_counts=True)
            labels.append(vals[np.argmax(counts)]) # majority class
            member_count.append(len(v))
        
        aggr_x = np.reshape(np.array(centroids), shape)
        aggr_y = np.array(labels)
        avg = np.mean(member_count)

        with open(path, 'wb') as f:
            pickle.dump((aggr_x, aggr_y, avg), f)

    return avg, shape, Dataset(data.name, aggr_x, data.x_test, aggr_y, data.y_test)



    
    