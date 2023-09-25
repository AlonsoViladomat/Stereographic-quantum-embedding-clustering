from audioop import avg
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random as rnd
import warnings

from functions_3D import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'
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

#Getting the received analog signal
rxsignal = data['rxsignal']
#print(rxsignal)       
#choosing first data column
rxsignal = rxsignal[:,0]


num_point = [320,640,1280,2560,3200,6400,12800,25600,52124]
radii = [0.01,0.1,0.25,0.5,1,1.5,2,5,10,100]

results = []

#stores results in a file
out_folder = os.path.join(cwd+ "/results")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

#stores results in a file
out_folder = os.path.join(cwd+ "/results/stereographic_sim_random_choice")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

#stores results in a file
out_folder = os.path.join(cwd+ "/results/stereographic_sim_random_choice/classical_robertoshit")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass


#print(len(rxsignal))
for num_points in num_point:
    for r in radii:

        # create array of random samples from 0:52124, sample from that and use that as the common indices to sample from rxsignal and bits 
        
        
        #print(index)
        #index.sort()

        #Alphabet Tx for initial centroid
        alph_x = (r*r)*2*alphabet.real/(r*r + np.absolute(alphabet)**2)
        alph_y = (r*r)*2*alphabet.imag/(r*r + np.absolute(alphabet)**2)
        alph_z = r*(-r*r + np.absolute(alphabet)**2)/(r*r + np.absolute(alphabet)**2)

        tx_alph = []
        for i in range(len(alphabet)):
            tx_alph.append([float(alph_x[i]),float(alph_y[i]),float(alph_z[i])])
        
        tx_alph = np.asarray(tx_alph)
        
        for iterator1 in range(100):
            avgAccuracy1 = 0
            avgAccuracy2 = 0
            avgNumIter = 0
            for iterator2 in range(100):
                
                index = rnd.sample(range(len(rxsignal)), num_points)
                #dataIn =[]
                #sampleLabelsIn = []
                dataIn = np.zeros(num_points)
                sampleLabelsIn = np.zeros(num_points)
                
                dataIn = rxsignal[index]
                sampleLabelsIn = sampleLabels[index]

                #print(dataIn)
                #print(sampleLabels)

                d_x = (r*r)*2*dataIn.real/(r*r + np.absolute(dataIn)*np.absolute(dataIn))
                d_y = (r*r)*2*dataIn.imag/(r*r + np.absolute(dataIn)*np.absolute(dataIn))
                d_z = r*(-r*r + np.absolute(dataIn)*np.absolute(dataIn))/(r*r + np.absolute(dataIn)*np.absolute(dataIn))

                

                tx_init_pt = []
                for i in range(len(dataIn)):
                    tx_init_pt.append([d_x[i],d_y[i],d_z[i]])
                tx_init_pt = np.asarray(tx_init_pt)

                
                dataClusters, centroids, numIter = classical_k_mean(64,tx_init_pt,tx_alph,50,'cartesian','fixed')

                accuracy1 = accuracy_score(sampleLabelsIn, dataClusters)
                accuracy2 = balanced_accuracy_score(sampleLabelsIn, dataClusters)
                
                avgAccuracy1 = avgAccuracy1 + accuracy1
                avgAccuracy2 = avgAccuracy2 + accuracy2
                avgNumIter = avgNumIter + numIter
                
            avgAccuracy1 = avgAccuracy1/1
            avgAccuracy2 = avgAccuracy2/1
            diff = abs(avgAccuracy1 - avgAccuracy2)
            avgNumIter = avgNumIter/100

            #appends results
            results.append([r,num_points,avgAccuracy1,avgAccuracy2,avgNumIter])
            print(r,num_points,avgAccuracy1,avgAccuracy2,diff,avgNumIter)

results = np.asarray(results)

#saves results array in a file
string = file.partition(".")[0]
output_file = "Results_classical_stereographic_random_choice_robertoMod_" + string
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)


print(results)



print('done')