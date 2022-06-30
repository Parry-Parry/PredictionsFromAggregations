import numpy as np
import collections
import pickle
from pathlib import Path, PurePath
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import tensorflow.keras as keras

from src.models.thesis_baseline.generate_baseline_model import gen_model
from src.models.thesis_baseline.epsilon_neighbours import *
from src.models.thesis_baseline.gaussian import *

### DATA FORMATTING ###

def flatten(x, num_instances=0):
    """
    Convert n x m image to vector
    """
    x_vecs = []
    n = num_instances
    if n==0:
        n = x.shape[0]
    for i in range(n):
        v = x[i].reshape(1,-1)
        x_vecs.append(v[0])
    return np.array(x_vecs)

def dataset_normalize(data):
    (x_train, y_train), (x_test, y_test) = data
    
    x_train = x_train / np.float32(255)
    x_test = x_test / np.float32(255)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    dataset = {
        'x_train' : x_train,
        'y_train' : y_train,
        'x_test' : x_test,
        'y_test' : y_test
    }

    return dataset

### INITIAL CLUSTERS ###

def partition(x_input_vecs, num_clusters, SEED, write_path=""):
    """
    K-Means partition and storing the partition details
    """
    x = []
    y = []
        
    # Run K-means and save the vecs/cluster ids in a tsv file
    cluster_ids = KMeans(n_clusters=num_clusters, random_state=SEED).fit_predict(x_input_vecs)
    with open(write_path, "w") as f:
        for i in range(len(x_input_vecs)):            
            x_i = x_input_vecs[i]
            c_i = cluster_ids[i]
            line = ''.join(str(e)+' ' for e in x_i)
            f.write(line + " " + str(c_i) + "\n")
            x.append(x_i)
            y.append(c_i)
    
    return (x, y)

def groupClusters(x, y):
    cluster_members =  collections.defaultdict(list)
    for a, b in zip(x, y): cluster_members[b].append(a)
    return cluster_members

def groupLabels(x, y, y_cluster_id, num_clusters, dataset_name):
    cluster_labels = collections.defaultdict(list)
    prob_cluster_labels = []
    
    num_labels = len(np.unique(y))
    if 'CIFAR' in dataset_name:
        for i in range(len(x)): cluster_labels[(y_cluster_id[i])].append(y[i][0]) # Strip Array wrapper for counter
    else:
        for i in range(len(x)): cluster_labels[(y_cluster_id[i])].append(y[i])

    for k in range(num_clusters):
        local_freqs = dict(collections.Counter(cluster_labels[k]))        
        values = []
        
        values = [local_freqs[i] if i in local_freqs else 0 for i in range(num_labels)]           
            
        nvalues = [float(i)/sum(values) for i in values]        
        prob_cluster_labels.append(nvalues)
    return prob_cluster_labels

### EXPERIMENT FUNCTIONS ###

def run_model(X : tuple, Y : tuple, shape : tuple, n_classes : int, params : dict, log_data : dict):
    '''
    Create a model with 'shape' input and 'n_classes' output, record performance.
    '''
    
    x, y, z = shape

    x_train, x_test = X
    y_train, y_test = Y

    x_train = x_train.reshape(len(x_train), x, y, z)

    out={}

    model = gen_model(shape, n_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=params['batch_size'], validation_split=0.2, epochs=params['epochs'], verbose=0)

    if params['save_history']:
        name = log_data['Algorithm'] + log_data['Dataset'] + log_data['K Value'] + log_data['Epsilon Value'] + ".pkl"
        with open(PurePath(params['path'], name), 'wb') as f:
            pickle.dump(history.history, f)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    out['accuracy'] = accuracy_score(y_test, y_pred)
    out['precision'] = precision_score(y_test, y_pred , average="weighted")
    out['recall'] = recall_score(y_test, y_pred , average="weighted")
    out['f1'] = f1_score(y_test, y_pred , average="weighted")

    out['algorithim'] = log_data['Algorithm']

    print("Accuracy on {} : {}".format(out['algorithim'], out['accuracy']))

    return out

def runTest(K : int, epsilon : float, X : tuple, Y : tuple, partitions : tuple, shape : tuple, params : dict, dataset_name : str, experiment : tuple):
    
    log_gauss = {
            'Algorithm' : 'Gaussian_Neighbourhood',
            'Dataset' : dataset_name,
            'K Value' : str(K),
            'Epsilon Value' : str(epsilon)
        }
    log_eps = {
            'Algorithm' : 'Epsilon_Neighbourhood',
            'Dataset' : dataset_name,
            'K Value' : str(K),
            'Epsilon Value' : str(epsilon)
        }
    log_complete = {
        'Algorithm' : 'Complete_Information',
        'Dataset' : dataset_name,
        'K Value' : str(K),
        'Epsilon Value' : str(epsilon)
    }

    results = {}

    gauss, eps, complete = experiment

    x_train, x_test = X
    y_train, y_test = Y

    x, y = partitions
    
    d1, d2, d3 = shape

    x_test = x_test.reshape(len(x_test), d1, d2, d3)

    members = groupClusters(x, y)

    sigma = np.eye(d1 * d2 * d3)
    mu = {k : np.mean(v, axis=0).flatten() for k, v in members.items()}

    n_classes=len(np.unique(y_train))
    prob_cluster_labels = groupLabels(x, y_train, y, K, dataset_name)

    ### GAUSS CALCULATIONS ###

    if gauss:

        x_gauss, y_gauss = reconstructWithGaussians(mu, sigma, members, prob_cluster_labels, n_classes)

        x_gauss = x_gauss.reshape(len(x_gauss),d1,d2,d3)
        y_gauss = keras.utils.to_categorical(y_gauss, n_classes)

        results['gaussian'] = run_model((x_gauss, x_test), (y_gauss, y_test), shape, n_classes, params, log_gauss)

    ### EPSILON NEIGHBOURHOOD CALCULATIONS ###

    if eps:

        x_eps, y_eps = reconstructWithEpsilonNeighborhood(mu, members, prob_cluster_labels, n_classes, epsilon)

        x_eps = np.array(x_eps)
        y_eps = np.array([point[0] for point in y_eps])

        x_eps = x_eps.reshape(len(x_eps),d1,d2,d3)
        y_eps = keras.utils.to_categorical(y_eps, n_classes)

        results['epsilon'] = run_model((x_eps, x_test), (y_eps, y_test), shape, n_classes, params, log_eps)

    ### COMPLETE INFORMATION CALCULATIONS ###

    if complete:

        x_train = x_train.reshape(len(x_train),d1,d2,d3)
        y_train = keras.utils.to_categorical(y_train, n_classes)

        results['complete'] = run_model((x_train, x_test), (y_train, y_test), shape, n_classes, params, log_complete)

    return results


