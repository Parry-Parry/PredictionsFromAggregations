import numpy as np


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
    num_clusters = len(mu.keys())
    
    for i in range(num_clusters):
        mu_i = mu[i] # ith cluster centroid
        p_label_dist = label_info[i]
        nmembers = len(cluster_info[i]) 
        sampled_vecs = sampleFromNeighborhood(mu_i, nmembers, epsilon)
        
        for s in sampled_vecs:
            x_vecs.append(s) # random point
            y_vecs.append(np.random.choice(num_labels, 1, p=p_label_dist)) # random label
    
    return x_vecs, y_vecs   