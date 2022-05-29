import numpy as np
import collections
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import keras

from generate_baseline_model import gen_model

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
        x_vecs.append(v[0]/255.0)
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

def runKmeans1(K : int, X : tuple, y : tuple, shape : tuple, params : dict) -> dict:
    out = {}

    d1, d2, d3 = shape

    x_train, x_test = X
    y_train, y_test = y

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

    model = gen_model(shape, n_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x2_train, y2_train, batch_size=params['batch_size'], validation_split=0.2, epochs=params['epochs'], verbose=2)
    out['history'] = history

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    out['accuracy'] = accuracy_score(y_test, y_pred)
    out['precision'] = precision_score(y_test, y_pred , average="weighted")
    out['recall'] = recall_score(y_test, y_pred , average="weighted")
    out['f1'] = f1_score(y_test, y_pred , average="weighted")

    return out




def partition(x_input_vecs, num_clusters, SEED, path="", write_path=""):
    """
    K-Means partition and storing the partition details
    """
    x = []
    y = []
        
    if path:
        # file exists        
        with open(path) as f:
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
        # Run K-means and save the vecs/cluster ids in a tsv file
        cluster_ids = KMeans(n_clusters=num_clusters, random_state=SEED).fit_predict(x_input_vecs)
        for i in range(len(x_input_vecs)):            
            x_i = x_input_vecs[i]
            c_i = cluster_ids[i]
            
            x.append(x_i)
            y.append(c_i)

        if write_path: # TODO : Write to file
            with open(write_path, "w") as f:
                line = ''.join(str(e)+' ' for e in x_i)
                f.write(line + " " + str(c_i) + "\n")
    
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

def computeGaussianParameters(cluster_members, num_clusters, dimension):
    mean_vecs = np.zeros((num_clusters, dimension), np.float32)
    std_vecs = np.zeros((num_clusters, dimension), np.float32)
    for k in range(K):
        for j in range(dimension):
            key = str(k) + ':' + str(j)
            if key in cluster_members:
                vals = np.array(cluster_members[key])
                mean_vecs[k][j] = np.mean(vals)
                std_vecs[k][j] = np.std(vals)
    
    return mean_vecs, std_vecs

def sampleFromGaussian(mu, sigma, num_samples):
    vecs = []    
    for i in range(num_samples):
        vecs.append(np.random.normal(mu, sigma))        
    return vecs    

def reconstructWithGaussians(mu, sigma, cluster_info, label_info, num_labels):
    x_vecs = []
    y_vecs = []
    num_clusters = mu.shape[0]
    
    for i in range(num_clusters):
        mu_i = mu[i] # ith cluster centroid
        sigma_i = sigma[i] # std dev vector of ith cluster centroid
        
        p_label_dist = label_info[i]
        nmembers = len(cluster_info[str(i) + ':0']) # each dimension will have identical nmembers        
        sampled_vecs = sampleFromGaussian(mu_i, sigma_i, nmembers)
        
        for s in sampled_vecs:
            x_vecs.append(s) # random point  
            y_vecs.append(np.random.choice(num_labels, 1, p=p_label_dist)) # random label
            
    return x_vecs, y_vecs

def sampleFromNeighborhood(x, num_samples, epsilon):
    vecs = []
    d = x.shape[0]
    limit = epsilon/float(d)
    
    for k in range(num_samples):
        vec = np.zeros(d)
        for i in range(d):
            x_i = x[i]
            vec[i] = np.maximum(0, np.random.uniform(x_i-epsilon, x_i+epsilon))
        vecs.append(vec)
    return vecs 

def reconstructWithEpsilonNeighborhood(mu, cluster_info, label_info, num_labels, epsilon):
    x_vecs = []
    y_vecs = []
    num_clusters = mu.shape[0]
    
    for i in range(num_clusters):
        mu_i = mu[i] # ith cluster centroid
        p_label_dist = label_info[i]
        nmembers = len(cluster_info[str(i) + ':0']) # each dimension will have identical nmembers        
        sampled_vecs = sampleFromNeighborhood(mu_i, nmembers, epsilon)
        
        for s in sampled_vecs:
            x_vecs.append(s) # random point
            y_vecs.append(np.random.choice(num_labels, 1, p=p_label_dist)) # random label
    
    return x_vecs, y_vecs   