from audioop import avg
import os
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import random as rnd
import warnings
from timeit import default_timer as timer

from functions_3D_new_updation import *
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
alphabet = scipy.io.loadmat(file_name_2)
alphabet = alphabet['alphabet']

#Loading the real world dataset 
data = scipy.io.loadmat(file_name)


#Getting initial transmission labels and converting to decimal
initial_bits = data["bits"]
initial_bits = bit_to_decimal(initial_bits)
sampleLabels = np.add(initial_bits,1)
sampleLabels = np.column_stack([sampleLabels,sampleLabels,sampleLabels,sampleLabels,sampleLabels])
sampleLabels = np.reshape(sampleLabels,(260620,1))

#Getting the received analog signal
rxsignal = data['rxsignal']
#print(rxsignal)       
#choosing first data column
# print(rxsignal.shape)

# Using all data columns 
rxsignal = np.reshape(rxsignal, (260620,1))



num_point = [3200,6400,12800,25600,51200,76800,102400,153600, 204800, 256000]
radii = [1,2,2.5,3,3.5,4,4.5,5,6,7.5,10]

results = []

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
out_folder = os.path.join(cwd+ "/results/overfitting/all_points/stereographic")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass



for num_points in num_point:
    
    for r in radii:

    
        
        avgAccuracy_train = 0
        #avgAccuracy2 = 0
        avgNumIter_train = 0
        avgTime_train = 0

        avgAccuracy_test = 0
        #avgAccuracy2 = 0
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

            #print(dataIn_train , " blah " , dataIn_test)


            #### TRAINING ####

            #Starting time measurement for training
            start = timer()

            #Transforming dataset to ISP
            d_x = (r*r)*2*dataIn_train.real/(r*r + np.absolute(dataIn_train)*np.absolute(dataIn_train))
            d_y = (r*r)*2*dataIn_train.imag/(r*r + np.absolute(dataIn_train)*np.absolute(dataIn_train))
            d_z = r*(-r*r + np.absolute(dataIn_train)*np.absolute(dataIn_train))/(r*r + np.absolute(dataIn_train)*np.absolute(dataIn_train))

            
            #Alphabet ISP for initial centroids                
            alph_x = (r*r)*2*alphabet.real/(r*r + np.absolute(alphabet)**2)
            alph_y = (r*r)*2*alphabet.imag/(r*r + np.absolute(alphabet)**2)
            alph_z = r*(-r*r + np.absolute(alphabet)**2)/(r*r + np.absolute(alphabet)**2)
            
            #Formatting alphabet points
            tx_alph = np.column_stack((alph_x,alph_y,alph_z))
            # for i in range(len(alphabet)):
            #     tx_alph.append([float(alph_x[i]),float(alph_y[i]),float(alph_z[i])])
            
            # tx_alph = np.asarray(tx_alph)
            
            #Formatting data points
            tx_init_pt = np.column_stack((d_x, d_y, d_z))
            # for i in range(len(dataIn_train)):
            #     tx_init_pt.append([d_x[i],d_y[i],d_z[i]])
            # tx_init_pt = np.asarray(tx_init_pt)


            #Performing clustering
            dataClusters_train, centroids_train, numIter_train = classical_k_mean(64,tx_init_pt,tx_alph,50,r)

            #Ending time measurement
            end = timer()

            #centroids_train.shape 

            #centroids_train[:, [0,1,2]].shape 

            accuracy_train = accuracy_score(sampleLabelsIn_train, dataClusters_train)
            #accuracy2 = balanced_accuracy_score(sampleLabelsIn, dataClusters)
            t_train = end - start
            
            avgAccuracy_train = avgAccuracy_train + accuracy_train
            #avgAccuracy2 = avgAccuracy2 + accuracy2
            avgNumIter_train = avgNumIter_train + numIter_train
            avgTime_train = avgTime_train + t_train 

            # print(centroids_train.shape)
            # centroids_train_in = np.delete(centroids_train,3,1)     
            # print(centroids_train_in.shape)

            centroids_train_in = np.delete(centroids_train,3,1)
            #### TESTING ####
            
            #Starting time measurement for training
            start = timer()

            #Transforming dataset to ISP
            d_x = (r*r)*2*dataIn_test.real/(r*r + np.absolute(dataIn_test)*np.absolute(dataIn_test))
            d_y = (r*r)*2*dataIn_test.imag/(r*r + np.absolute(dataIn_test)*np.absolute(dataIn_test))
            d_z = r*(-r*r + np.absolute(dataIn_test)*np.absolute(dataIn_test))/(r*r + np.absolute(dataIn_test)*np.absolute(dataIn_test))

            
            #Alphabet ISP for initial centroids                
            # Here 'alphabet' is the values of the centroids received from training data -- centroids_train
            
            #Formatting alphabet points
            # tx_alph = []
            # for i in range(len(alphabet)):
            #     tx_alph.append([float(alph_x[i]),float(alph_y[i]),float(alph_z[i])])
            
            # tx_alph = np.asarray(tx_alph)
            
            #Formatting data points
            tx_init_pt = np.column_stack((d_x, d_y, d_z))
            # for i in range(len(dataIn_train)):
            #     tx_init_pt.append([d_x[i],d_y[i],d_z[i]])
            # tx_init_pt = np.asarray(tx_init_pt)

            # print(centroids_train.shape)
            # print(centroids_train_in.shape)
            # print(centroids_test.shape)
            #Performing nearest mean classification using training-obtained centroids as centroids
            #dataClusters_test, centroids_test, numIter_test = classical_k_mean(64,tx_init_pt,centroids_train_in,50)

            # #Performing simple classsification using training-obtained centroids as initital centroids
            dataClusters_test, centroids_test, numIter_test = classical_k_mean(64,tx_init_pt,centroids_train[:,[0,1,2]],1,r)

            #Ending time measurement
            end = timer()

            accuracy_test = accuracy_score(sampleLabelsIn_test, dataClusters_test)
            #accuracy2 = balanced_accuracy_score(sampleLabelsIn, dataClusters)
            t_test = end - start
            
            avgAccuracy_test = avgAccuracy_test + accuracy_test
            #avgAccuracy2 = avgAccuracy2 + accuracy2
            avgNumIter_test = avgNumIter_test + numIter_test
            avgTime_test = avgTime_test + t_test 





            
        #avgAccuracy_train = avgAccuracy_train/1
        #avgAccuracy2 = avgAccuracy2/1
        #diff = abs(avgAccuracy1 - avgAccuracy2)
        avgNumIter_train= avgNumIter_train/100
        avgTime_train = avgTime_train/100

        #avgAccuracy_test = avgAccuracy_test/1
        #avgAccuracy2 = avgAccuracy2/1
        #diff = abs(avgAccuracy1 - avgAccuracy2)
        avgNumIter_test= avgNumIter_test/100
        avgTime_test = avgTime_test/100

        #Difference in training and testing accuracy, seeing if any overfitting occured
        diff_acc = avgAccuracy_test - avgAccuracy_train
        #appends results
        results.append([r , num_points , diff_acc, avgAccuracy_train , avgAccuracy_test, avgNumIter_train , avgNumIter_test , avgTime_train , avgTime_test])
        print(r , num_points , diff_acc, avgAccuracy_train , avgAccuracy_test, avgNumIter_train , avgNumIter_test , avgTime_train , avgTime_test)



results = np.asarray(results)

#saves results array in a file
string = file.partition(".")[0]
output_file = "all_points_new_update_overfitting_stereographic_results_" + string
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)


print(results)



print('done')
