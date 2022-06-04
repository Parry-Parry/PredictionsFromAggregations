import numpy as np
import collections
import pickle
from pathlib import Path, PurePath
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import keras

from generate_baseline_model import gen_model
from src.models.baseline.epsilon_neighbours import *
from src.models.baseline.gaussian import *

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

def dataset_normalize(train, test):
    x_train, y_train = train
    x_test, y_test = test

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

def groupData(x, y):
    cluster_members = {} # indexed by cluster-id, dimension 
    for i in range(len(x)):
        x_i = x[i]
        y_i = y[i]
        for j in range(len(x_i)):
            key = str(y_i) + ':' + str(j)
            if not key in cluster_members:
                cluster_members[key] = []
            cluster_members[key].append(x_i[j])
    
    return cluster_members

def groupClusters(x, y):
    cluster_members =  collections.defaultdict(list)
    for a, b in zip(x, y): cluster_members[b].append(a)
    return dict(sorted(cluster_members))
'''
def groupLabels(x, y, y_cluster_id, num_clusters, num_labels):
    cluster_labels = {} # indexed by cluster-id, dimension 
    prob_cluster_labels = []
    
    for i in range(len(x)):
        key = str(y_cluster_id[i])
        if not key in cluster_labels:
            cluster_labels[key] = []        
        cluster_labels[key].append(y[i]) # append the ground truth label for this cluster id

    for k in range(num_clusters):
        key = str(k)
        local_freqs = dict(collections.Counter(cluster_labels[key]))        
        values = []
        
        for i in range(num_labels):
            f = 0
            if i in local_freqs:
                f = local_freqs[i]
            values.append(f)                
            
        nvalues = [float(i)/sum(values) for i in values]        
        prob_cluster_labels.append(nvalues)
        
    return prob_cluster_labels
'''
def groupLabels(x, y, y_cluster_id, num_clusters): # TODO : Fix this rank mess
    cluster_labels = collections.defaultdict(list)
    prob_cluster_labels = []
    
    num_labels = len(np.unique(y))

    for i in range(len(x)): cluster_labels[(y_cluster_id[i])].append(y[i]) 

    for k in range(num_clusters):
        local_freqs = dict(collections.Counter(cluster_labels[k]))        
        values = []
        
        values = [local_freqs[i] for i in range(num_labels) if i in local_freqs]           
            
        nvalues = [float(i)/sum(values) for i in values]        
        prob_cluster_labels.append(nvalues)
        
    return prob_cluster_labels

### EXPERIMENT FUNCTIONS ###

def run_model(X : tuple, Y : tuple, shape : tuple, n_classes : int, params : dict, log_data : dict):
    '''
    Create a model with 'shape' input and 'n_classes' output, record performance.
    '''

    x_train, x_test = X
    y_train, y_test = Y

    out={}

    model = gen_model(shape, n_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=params['batch_size'], validation_split=0.2, epochs=params['epochs'], verbose=2)

    if params['save_history']:
        name = log_data['Algorithm'] + log_data['K'] + log_data['epsilon'] + ".pkl"
        with open(PurePath(params['path'], 'history', name)) as f:
            pickle.dump(history, f)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    out['accuracy'] = accuracy_score(y_test, y_pred)
    out['precision'] = precision_score(y_test, y_pred , average="weighted")
    out['recall'] = recall_score(y_test, y_pred , average="weighted")
    out['f1'] = f1_score(y_test, y_pred , average="weighted")

    out['algorithim'] = log_data['Algorithm']

    return out

def runTest(K : int, epsilon : float, X : tuple, Y : tuple, partitions : tuple, shape : tuple, params : dict):
    
    log_gauss = {
            'Algorithm' : 'Gaussian_Neighbourhood',
            'K Value' : str(K),
            'Epsilon Value' : str(epsilon)
        }
    log_eps = {
            'Algorithm' : 'Epsilon_Neighbourhood',
            'K Value' : str(K),
            'Epsilon Value' : str(epsilon)
        }
    log_complete = {
        'Algorithm' : 'Complete_Information',
        'K Value' : str(K),
        'Epsilon Value' : str(epsilon)
    }

    x_train, x_test = X
    y_train, y_test = Y

    x, y = partitions
    members = groupData(x, y)

    n_classes=len(np.unique(y_train))
    prob_cluster_labels = groupLabels(x, y_train, y, K)

    mu, sigma = computeGaussianParameters(members, K, len(x[0]))

    ### GAUSS CALCULATIONS ###

    x_gauss, y_gauss = reconstructWithGaussians(mu, sigma, members, prob_cluster_labels, n_classes)

    x_gauss = np.expand_dims(x_gauss, -1)
    x_test = np.expand_dims(x_test, -1)

    y_gauss = keras.utils.to_categorical(y_gauss, n_classes)

    gauss_out = run_model((x_gauss, x_test), (y_gauss, y_test), shape, n_classes, params, log_gauss)

    ### EPSILON NEIGHBOURHOOD CALCULATIONS ###

    x_eps, y_eps = reconstructWithEpsilonNeighborhood(mu, members, prob_cluster_labels, n_classes, epsilon)

    x_eps = np.array(x_eps)
    y_eps = np.array([point[0] for point in y_eps])

    x_eps = np.expand_dims(x_eps, -1)

    y_eps = keras.utils.to_categorical(y_eps, n_classes)

    eps_out = run_model((x_eps, x_test), (y_eps, y_test), shape, n_classes, params, log_eps)

    ### COMPLETE INFORMATION CALCULATIONS ###

    x_train = np.expand_dims(x_train, -1)

    y_train = keras.utils.to_categorical(y_train, n_classes)

    complete_out = run_model((x_train, x_test), (y_train, y_test), shape, n_classes, params, log_complete)

    return gauss_out, eps_out, complete_out 


