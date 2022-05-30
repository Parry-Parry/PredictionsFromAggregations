import numpy as np
import keras
from sklearn.cluster import MiniBatchKMeans
from src.models.baseline.helper import run_model


def infer_cluster_labels(kmeans, actual_labels: np.array):
    inferred_labels = {}
    for i in range(kmeans.n_clusters):
        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)
        # append actual labels for each point in cluster
        labels.append(actual_labels[index])
        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))
        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]    
    return inferred_labels  

def runKmeans(K : int, X : tuple, Y : tuple, shape : tuple, params : dict) -> dict:
    '''
    Run Baseline and Test with Convnet
    '''
    log_data = {
            'Algorithm' : 'K-Means',
            'K Value' : str(K),
            'Epsilon Value' : 'NA'
        }
        

    d1, d2, d3 = shape

    x_train, x_test = X
    y_train, y_test = Y

    kmeans = MiniBatchKMeans(K)
    kmeans.fit(x_train)

    cluster_labels = infer_cluster_labels(kmeans, y_train)

    labels = []

    for i in range(K):
        for k, v in cluster_labels.items():
            if i in v: labels.append(k)
    
    y_tmp = np.array(labels)

    centroids = kmeans.cluster_centers_.reshape(K,d1,d2)

    n_classes = len(np.unique(y_train))

    x2_train = np.expand_dims(centroids, -1)
    y2_train =  keras.utils.to_categorical(y_tmp, n_classes)

    return run_model((x2_train, x_test), (y2_train, y_test), shape, n_classes, params, log_data)