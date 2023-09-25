from audioop import avg
import os
import numpy as np
import scipy.io
import random as rnd
import warnings
from timeit import default_timer as timer

from functions_3D_new_update_stopping_criteria import *
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat'
file_name = os.path.join(cwd+"/realdata", file)
file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

#Loading the alphabet - initial analog transmission values
alphabet = scipy.io.loadmat(file_name_2)['alphabet']

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
radii = [0.01,0.1,0.5,1,1.5,2,2.25,2.5,2.75,3,3.5,4,4.5,5,10,100]
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
out_folder = os.path.join(cwd+ "/results/stopping_criteria/stereographic_sim_random_choice")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


for num_points in num_point:
    
    for r in radii:

        for iterator1 in range(100):
            
            avgAccuracy = np.zeros(maxNumIter)
            prob_cluster_change = np.zeros(maxNumIter)

            for iterator2 in range(100):

                
                
                index = rnd.sample(range(len(rxsignal)), num_points)

                dataIn = np.zeros(num_points)
                sampleLabelsIn = np.zeros(num_points)
                
                dataIn = rxsignal[index]
                sampleLabelsIn = sampleLabels[index]

                # #Starting time measurement
                # start = timer()

                #Transforming dataset to ISP
                d_x = (r*r)*2*dataIn.real/(r*r + np.absolute(dataIn)*np.absolute(dataIn))
                d_y = (r*r)*2*dataIn.imag/(r*r + np.absolute(dataIn)*np.absolute(dataIn))
                d_z = r*(-r*r + np.absolute(dataIn)*np.absolute(dataIn))/(r*r + np.absolute(dataIn)*np.absolute(dataIn))

                
                #Alphabet ISP for initial centroids                
                alph_x = (r*r)*2*alphabet.real/(r*r + np.absolute(alphabet)**2)
                alph_y = (r*r)*2*alphabet.imag/(r*r + np.absolute(alphabet)**2)
                alph_z = r*(-r*r + np.absolute(alphabet)**2)/(r*r + np.absolute(alphabet)**2)
                
                # #Formatting alphabet points
                
                tx_alph = np.column_stack((alph_x,alph_y,alph_z))
                
                # #Formatting alphabet points
                
                tx_init_pt =  np.column_stack((d_x,d_y,d_z))

                #Performing clustering 
                dataClusters, centroids, numIter, accuracy, cluster_change = classical_k_mean(64,tx_init_pt,tx_alph,maxNumIter,r,sampleLabelsIn)

                # #Ending time measurement
                # end = timer()

                # accuracy = accuracy_score(sampleLabelsIn, dataClusters)
                #accuracy2 = balanced_accuracy_score(sampleLabelsIn, dataClusters)
                # t = end - start
                
                avgAccuracy = avgAccuracy + accuracy
                prob_cluster_change = prob_cluster_change + cluster_change
                
            avgAccuracy = avgAccuracy/1
            prob_cluster_change = prob_cluster_change / 100

            # avgAccuracyToAppend = '\t'.join(avgAccuracy)
            # prob_cluster_changeToAppend = '\t'.join(prob_cluster_change)
            #appends results
            # results.append([r,num_points,avgAccuracyToAppend,prob_cluster_changeToAppend])
            # print(r,num_points,avgAccuracyToAppend,prob_cluster_changeToAppend)

            # appends results
            arrtoappend = np.append(     np.append(	np.array([r,num_points]), avgAccuracy	),	prob_cluster_change)
            results.append(arrtoappend)
            print(r,num_points,avgAccuracy,prob_cluster_change)
            #print(arrtoappend)

results = np.asarray(results)

#saves results array in a file
string = file.partition(".")[0]
output_file = "stopping_criteria_expt_" + string
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)


print(results)



print('done')