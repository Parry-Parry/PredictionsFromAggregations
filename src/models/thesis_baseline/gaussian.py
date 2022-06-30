import numpy as np
import collections


def computeGaussianParameters(cluster_members, K, dimension): # TODO : Convert to multivariate normal
    mean_vecs = np.zeros((K, dimension), np.float32)
    std_vecs = np.zeros((K, dimension), np.float32)
    for k in range(K):
        for j in range(dimension):
            key = str(k) + ':' + str(j)
            if key in cluster_members:
                vals = np.array(cluster_members[key])
                mean_vecs[k][j] = np.mean(vals)
                std_vecs[k][j] = np.std(vals)
    
    return mean_vecs, std_vecs

def computeMultivariateGaussianParameters(cluster_members): 
    mu = {}
    sigma = {}
    for k, v in cluster_members.items():
        X = np.stack(v, axis=0)
        mu[k] = np.mean(v, axis=0).flatten()
        sigma[k] = np.cov(X.T)
    return mu, sigma

def reconstructWithGaussians(mu, sigma, cluster_info, label_info, num_labels):
    x_vecs = []
    y_vecs = []
    num_clusters = len(mu.keys())
    
    for i in range(num_clusters):
        mu_i = mu[i] # ith cluster centroid
        sigma_i = sigma[i] # std dev vector of ith cluster centroid
        
        p_label_dist = label_info[i]
        nmembers = len(cluster_info[i]) # each dimension will have identical nmembers        
        sampled_vecs = np.random.multivariate_normal(mu_i, sigma_i, size=nmembers, check_valid='warn', tol=1e-8) 
        
        for s in sampled_vecs:
            x_vecs.append(s) # random point  
            y_vecs.append(np.random.choice(num_labels, 1, p=p_label_dist)) # random label
            
    return np.array(x_vecs), np.array([point[0] for point in y_vecs])