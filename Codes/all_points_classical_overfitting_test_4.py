import os
import numpy as np
import scipy.io
import random as rnd
import warnings
import math

from sklearn.metrics import accuracy_score
#from sklearn.metrics import balanced_accuracy_score
from functions_mod_classical_timing_sim import *
#from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

warnings.filterwarnings('ignore')
# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'
file_name = os.path.join(cwd+"/realdata", file)
file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')


num_point = [3200,6400,12800,25600,51200,76800,102400,153600, 204800, 256000]

results = []

#loads fixed centroids
alphabet = scipy.io.loadmat(file_name_2)['alphabet']
#print(alphabet.shape)
# alphabet = np.reshape(alphabet,alphabet.size)
# print(alphabet.shape)


# initial_bits = scipy.io.loadmat(file_name)["bits"]
# initial_bits = bit_to_decimal(initial_bits)
# sampleLabels = np.add(initial_bits,1)

#Loading the real world dataset 
data = scipy.io.loadmat(file_name)

#Getting initial transmission labels and converting to decimal
initial_bits = data["bits"]
initial_bits = bit_to_decimal(initial_bits)
sampleLabels = np.add(initial_bits,1)
sampleLabels = np.column_stack([sampleLabels,sampleLabels,sampleLabels,sampleLabels,sampleLabels])
sampleLabels = np.reshape(sampleLabels,(260620,1))

#data = scipy.io.loadmat(file_name)
rxsignal = data['rxsignal']

# Using all data columns 
rxsignal = np.reshape(rxsignal, (260620,1))
#rxsignal = rxsignal[:,0]

#stores results in a file
out_folder = os.path.join(cwd+ "/results")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

#stores results in a file
out_folder = os.path.join(cwd+ "/results/overfitting")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


#stores results in a file
out_folder = os.path.join(cwd+ "/results/overfitting/all_points")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


#stores results in a file
out_folder = os.path.join(cwd+ "/results/overfitting/classical_sim_random_choice")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass



for num_points in num_point:

    

    avgAccuracy_train = 0
    avgNumIter_train = 0
    avgTime_train = 0

    avgAccuracy_test = 0
    avgNumIter_test = 0
    avgTime_test = 0

    for iterator2 in range(100):

        #choosing the indices randomly 
        index = rnd.sample(range(len(rxsignal)), num_points)
        
        #choosing to split the array into 80% training and 20% testing data
        split = math.floor(num_points * 4 / 5)

        #indiced of the elements for training and testing datasets
        train_index = index[0:split]
        test_index = index[split:]
        
        
        #initialising training and test data arrays
        dataIn_train = np.zeros(split)
        sampleLabelsIn_train = np.zeros(split)

        dataIn_test = np.zeros((num_points - split))
        sampleLabelsIn_test = np.zeros((num_points - split))

        

        #creating training data array
        dataIn_train = rxsignal[train_index]
        sampleLabelsIn_train = sampleLabels[train_index]
        

        #creating test data array
        dataIn_test = rxsignal[test_index]
        sampleLabelsIn_test = sampleLabels[test_index]
        


        #### TRAINING ####

        #Starting time measurement for training
        start = timer()

        #print(dataIn_train.shape)
        dataIn_train = np.column_stack([dataIn_train.real, dataIn_train.imag])
        #print(dataIn_train.shape)
        #print(alphabet.shape)
        alphabet_in = np.column_stack([alphabet.real, alphabet.imag])
        #print(alphabet.shape)
        dataClusters_train, centroids_train, numIter_train = classical_k_mean(64,dataIn_train,alphabet_in,50)

        #Ending time measurement for training
        end = timer() 

        accuracy_train = accuracy_score(sampleLabelsIn_train, dataClusters_train)
        t_train = end - start
        
        avgAccuracy_train = avgAccuracy_train + accuracy_train
        avgNumIter_train = avgNumIter_train + numIter_train
        avgTime_train = avgTime_train + t_train


        ### TESTING ###


        #Starting time measurement for training
        start = timer()

        dataIn_test = np.column_stack([dataIn_test.real, dataIn_test.imag])

        dataClusters_test, centroids_test, numIter_test = classical_k_mean(64,dataIn_test,centroids_train[:,[0,1]],1)
        
        #Ending time measurement
        end = timer()

        accuracy_test = accuracy_score(sampleLabelsIn_test, dataClusters_test)
        t_test = end - start
        
        avgAccuracy_test = avgAccuracy_test + accuracy_test
        avgNumIter_test = avgNumIter_test + numIter_test
        avgTime_test = avgTime_test + t_test 



    #avgAccuracy_train = avgAccuracy_train/1
    avgNumIter_train= avgNumIter_train/100
    avgTime_train = avgTime_train/100

    #avgAccuracy_test = avgAccuracy_test/1
    avgNumIter_test= avgNumIter_test/100
    avgTime_test = avgTime_test/100

    #Difference in training and testing accuracy, seeing if any overfitting occured
    diff_acc = avgAccuracy_test - avgAccuracy_train
    
    
    #appends results
    results.append([num_points , diff_acc, avgAccuracy_train , avgAccuracy_test, avgNumIter_train , avgNumIter_test , avgTime_train , avgTime_test])
    print(num_points , diff_acc, avgAccuracy_train , avgAccuracy_test, avgNumIter_train , avgNumIter_test , avgTime_train , avgTime_test)

        
results = np.asarray(results)


string = file.partition(".")[0]
output_file = "all_points_overfitting_test_classical_" + string 
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)

print(results)

print('done')