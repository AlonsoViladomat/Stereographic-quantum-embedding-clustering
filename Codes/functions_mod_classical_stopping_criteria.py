import os
import psutil
import seaborn as sns
import time
import numpy as np
import math as mt
from sklearn.metrics import accuracy_score

from multiprocessing import Pool

num_cpus = psutil.cpu_count(logical=True)


def classical_k_mean(k,data,fixed_centroids,maxNumIter, dataLabels):

    # Classical K-mean Algorithm
    
    # INPUTS:
    # k: int --- k of k-mean. Decides the initial number of centroids.
    # data: np.array --- 
    # fixed_centroids: np.array --- this array sets initial points as centroids (only usable for method = "fixed")
    # coordinateSystem: str --- 'polar' or 'cartesian', indicates the coordinate system used
    # maxNumIter: int --- Maximum number of iteration
    # method: str --- Method for assigning initial centroids 'random' or 'maxEstimate'

    
    # OUTPUTS:
    # dataClusters: np.array --- Output Clusters of the algorithm 
    
    # Clusters Before is used for understanding the changes between iterations
    dataClusters = np.zeros(len(data))
    clustersBefore = np.zeros(len(data))
    
    # For a point in centroids [x1,x2,x3] x1 and x2 are coordinates. x3 is the Centroid number ranging from 1 to
    k, whocares = fixed_centroids.shape
    
    centroids = np.zeros([k,3])
    centroids[:,2] = np.arange(1, k+1)
    centroids[:, [0, 1]] = fixed_centroids
        
    
    numIter = 0
    
    accuracy = np.zeros(maxNumIter)
    didClustersChange = np.zeros(maxNumIter)

    while numIter < maxNumIter:
        
        np.copyto(clustersBefore,dataClusters)
        
        # Assigning data points to different clusters according to nearest centroid.
        dataClusters = centroids[np.argmin(np.linalg.norm(np.tile(data,(len(centroids),1,1)) - np.reshape(centroids[:,[0,1]],(len(centroids),1,2)),axis = 2),axis = 0),2]
        
        # Uses normalized version of the polar data and assigns data points to nearest cenroids ????????????????????????

        
        if np.array_equal(dataClusters, clustersBefore, equal_nan=False):
            didClustersChange[numIter] = 1
        
        # Deleting empty cluster centroids
        delList = []
        for i in np.arange(len(centroids)): 
            if np.sum([dataClusters == centroids[i,2]]) == 0:
                delList.append(i)
        centroids = np.delete(centroids, delList, 0)

        # Finding centroids from data by taking the mean of points in the clusters restricting it on the surface of the sphere 
        for i in np.arange(len(centroids)):
            centroids[i,[0,1]] = data[dataClusters == centroids[i,2]].mean(axis = 0)      

        accuracy[numIter] = accuracy_score(dataLabels, dataClusters)              
        
        numIter = numIter + 1


    return dataClusters, centroids, numIter, accuracy, didClustersChange

def bit_to_decimal(data):
    #takes an array of bits and maps it to an array of decimals

    data_aux = []
    for x in data:
        aux = 0
        for k in range(len(x)):
            aux= aux + x[k]*2**((len(x)-1)-k)
    
        data_aux.append(aux)

    return data_aux

