from audioop import avg
import os
import numpy as np
import scipy.io
import random as rnd
import warnings
from timeit import default_timer as timer

from functions_mod_classical_stopping_criteria import *
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat'
file_name = os.path.join(cwd+"/realdata", file)
file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

#Loading the alphabet - initial analog transmission values
alphabet = scipy.io.loadmat(file_name_2)['alphabet']
# #Formatting alphabet points
tx_alph = np.column_stack([alphabet.real, alphabet.imag])

#Loading the real world dataset 
data = scipy.io.loadmat(file_name)

#Getting initial transmission labels and converting to decimal
initial_bits = data["bits"]
initial_bits = bit_to_decimal(initial_bits)
sampleLabels = np.add(initial_bits,1)

#Getting the received analog signal
rxsignal = data['rxsignal']
rxsignal = rxsignal[:,0]


num_point = [640,1280,2560,3200,6400,12800,25600,51200]
maxNumIter = 50

results = []

#stores results in a file
out_folder = os.path.join(cwd+ "/results")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

#stores results in a file
out_folder = os.path.join(cwd+ "/results/stopping_criteria")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

#stores results in a file
out_folder = os.path.join(cwd+ "/results/stopping_criteria/classical_sim_random_choice")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


for num_points in num_point:
    


    for iterator1 in range(100):
        
        avgAccuracy = np.zeros(maxNumIter)
        prob_cluster_change = np.zeros(maxNumIter)

        for iterator2 in range(100):

            
            
            index = rnd.sample(range(len(rxsignal)), num_points)

            dataIn = np.zeros(num_points)
            sampleLabelsIn = np.zeros(num_points)
            
            dataIn = rxsignal[index]
            sampleLabelsIn = sampleLabels[index]
            dataIn = np.column_stack([dataIn.real, dataIn.imag])


            #Performing clustering #def classical_k_mean(k,data,fixed_centroids,maxNumIter, dataLabels):
            dataClusters, centroids, numIter, accuracy, cluster_change = classical_k_mean(64,dataIn,tx_alph,maxNumIter,sampleLabelsIn)

            # #Ending time measurement
            # end = timer()

            # accuracy = accuracy_score(sampleLabelsIn, dataClusters)
            #accuracy2 = balanced_accuracy_score(sampleLabelsIn, dataClusters)
            # t = end - start
            
            avgAccuracy = avgAccuracy + accuracy
            prob_cluster_change = prob_cluster_change + cluster_change
            
        avgAccuracy = avgAccuracy/1
        prob_cluster_change = prob_cluster_change / 100

        
        # appends results
        arrtoappend = np.append(     np.append(	np.array([num_points]), avgAccuracy	),	prob_cluster_change)
        results.append(arrtoappend)
        print(num_points,avgAccuracy,prob_cluster_change)
        #print(arrtoappend)

results = np.asarray(results)

#saves results array in a file
string = file.partition(".")[0]
output_file = "classical_stopping_criteria_expt_" + string
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)


print(results)



print('done')