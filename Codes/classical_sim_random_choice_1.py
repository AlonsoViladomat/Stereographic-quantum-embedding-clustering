import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random as rnd
import warnings

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from functions import *
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

warnings.filterwarnings('ignore')
# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat'
file_name = os.path.join(cwd+"/realdata", file)
file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

num_point = [64,128,320,640,1280,2560,3200,6400,12800,25600,52124]

results = []

#loads fixed centroids
alphabet = scipy.io.loadmat(file_name_2)["alphabet"]
alphabet = np.reshape(alphabet,alphabet.size)
alphabet = np.column_stack([alphabet.real, alphabet.imag])

initial_bits = scipy.io.loadmat(file_name)["bits"]
initial_bits = bit_to_decimal(initial_bits)
sampleLabels = np.add(initial_bits,1)

data = scipy.io.loadmat(file_name)
rxsignal = data['rxsignal']
rxsignal = rxsignal[:,0]

#stores results in a file
out_folder = os.path.join(cwd+ "/results")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


#stores results in a file
out_folder = os.path.join(cwd+ "/results/classical_random_choice_with_time")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


for num_points in num_point:

    for iterator1 in range(100):

        avgAccuracy1 = 0
        avgAccuracy2 = 0
        avgNumIter = 0
        avgTime = 0

        for iterator2 in range(100):

            index = rnd.sample(range(len(rxsignal)), num_points)

            dataIn = rxsignal[index]
            sampleLabelsIn = sampleLabels[index]
            dataIn = np.column_stack([dataIn.real, dataIn.imag])

            start = timer()

            dataClusters, centroids, numIter = classical_k_mean(64,dataIn,alphabet,50,'cartesian','fixed')

            end = timer() 
    
            accuracy1 = accuracy_score(sampleLabelsIn, dataClusters)
            accuracy2 = balanced_accuracy_score(sampleLabelsIn, dataClusters)
            t = end - start 
    
            avgAccuracy1 = avgAccuracy1 + accuracy1
            avgAccuracy2 = avgAccuracy2 + accuracy2
            avgNumIter = avgNumIter + numIter
            avgTime = avgTime + t 
    
        avgAccuracy1 = avgAccuracy1/1
        avgAccuracy2 = avgAccuracy2/1
        #diff = abs(avgAccuracy1 - avgAccuracy2)
        avgNumIter = avgNumIter/100
        avgTime = avgTime/100

        #appends results
        results.append([num_points,avgAccuracy1,avgAccuracy2,avgNumIter,avgTime])
        print(num_points,avgAccuracy1,avgAccuracy2,avgNumIter,avgTime)

results = np.asarray(results)


string = file.partition(".")[0]
output_file = string + "_classical_random_choice_with_time" 
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)


print('done')