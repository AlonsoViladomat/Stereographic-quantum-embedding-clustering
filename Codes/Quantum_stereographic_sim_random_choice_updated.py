from audioop import avg
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random as rnd
import warnings
from timeit import default_timer as timer

from functions_quantum_3D_new import *
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
#from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat'
file_name = os.path.join(cwd+"/realdata", file)
file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

#Loading the alphabet - initial analog transmission values
alphabet = scipy.io.loadmat(file_name_2)['alphabet']
# alphabet = np.column_stack([alphabet.real, alphabet.imag])

#Loading the real world dataset 
data = scipy.io.loadmat(file_name)


#Getting initial transmission labels and converting to decimal
initial_bits = data["bits"]
initial_bits = bit_to_decimal(initial_bits)
sampleLabels = np.add(initial_bits,1)

#Getting the received analog signal
rxsignal = data['rxsignal']
#print(rxsignal)       
#choosing first data column
rxsignal = rxsignal[:,0]
# rxsignal = np.column_stack([rxsignal.real, rxsignal.imag])


num_point = [320,640,1280,2560,3200,6400,12800,25600,51200]
radii = [0.1,1,2,2.5,3,3.5,4,4.5,5,6,7.5,10,100]
shots = [5,10,20,50,100,200,500,1000,2000,5000,10000]

results = []

#stores results in a file
out_folder = os.path.join(cwd+ "/results")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


#stores results in a file
out_folder = os.path.join(cwd+ "/results/quantum_sims")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


#stores results in a file
out_folder = os.path.join(cwd+ "/results/quantum_sims/stereographic")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


for nshots in shots: 

    for num_points in num_point:
        
        for r in radii:

            for iterator1 in range(100):
                
                avgAccuracy = 0
                avgNumIter = 0
                avgTime = 0


                for iterator2 in range(100):

                    
                    index = rnd.sample(range(len(rxsignal)), num_points)
            
                    dataIn = np.zeros(num_points)
                    sampleLabelsIn = np.zeros(num_points)
                    
                    dataIn = rxsignal[index]
                    sampleLabelsIn = sampleLabels[index]

                    #Starting time measurement
                    start = timer()

                    
                    #Performing QUANTUM clustering --- def qk_mean(k, data, fixed_centroids, maxNumIter, r, shots):
                    dataClusters, centroids, numIter = qk_mean(64, dataIn, alphabet, 5, r, nshots)

                    #Ending time measurement
                    end = timer()

                    accuracy = accuracy_score(sampleLabelsIn, dataClusters)
                    print(accuracy)
                    t = end - start
                    
                    avgAccuracy = avgAccuracy + accuracy
                    avgNumIter = avgNumIter + numIter
                    avgTime = avgTime + t 
                    
                avgAccuracy = avgAccuracy/1
                avgNumIter = avgNumIter/100
                avgTime = avgTime/100

                #appends results
                results.append([r, num_points, nshots, avgAccuracy, avgNumIter, avgTime])
                print(r, num_points, nshots, avgAccuracy, avgNumIter, avgTime)

results = np.asarray(results)

#saves results array in a file
string = file.partition(".")[0]
output_file = "results_quantum_stereographic_" + string
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)


print(results)



print('done')