import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random as rnd

from functions_3D import *
from mpl_toolkits.mplot3d import Axes3D

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
       
#choosing first data column
rxsignal = rxsignal[:,0]


num_point = [320,640,1280,2560,3200,6400,12800,25600,52124]
radii = [0.01,0.1,0.25,0.5,1,1.5,2,5,10,100]

results = []


for num_points in num_point:
    for r in radii:

        # create array of random samples from 0:52124, sample from that and use that as the common indices to sample from rxsignal and bits 
        
        index = rnd.sample(range(52125), num_point)
        #index.sort()
        avgAccuracy = 0
        avgNumIter = 0
        
        for iterator in range(100):
            
            dataIn = []
            sampleLabelsIn = []

            for i in index:
                dataIn.append(rxsignal[i])
                sampleLabelsIn.append(sampleLabels[i])    

            d_x = (r*r)*2*dataIn.real/(r*r + np.absolute(dataIn)*np.absolute(dataIn))
            d_y = (r*r)*2*dataIn.imag/(r*r + np.absolute(dataIn)*np.absolute(dataIn))
            d_z = r*(-r*r + np.absolute(dataIn)*np.absolute(dataIn))/(r*r + np.absolute(dataIn)*np.absolute(dataIn))

            #Alphabet Tx for initial centroid
            alph_x = (r*r)*2*alphabet.real/(r*r + np.absolute(alphabet)**2)
            alph_y = (r*r)*2*alphabet.imag/(r*r + np.absolute(alphabet)**2)
            alph_z = r*(-r*r + np.absolute(alphabet)**2)/(r*r + np.absolute(alphabet)**2)

            tx_init_pt = []
            for i in range(len(dataIn)):
                    tx_init_pt.append([d_x[i],d_y[i],d_z[i]])
            tx_init_pt = np.asarray(tx_init_pt)

            tx_alph = []
            for i in range(len(alphabet)):
                    tx_alph.append([float(alph_x[i]),float(alph_y[i]),float(alph_z[i])])
            tx_alph = np.asarray(tx_alph)


            dataClusters, centroids, numIter = classical_k_mean(64,tx_init_pt,tx_alph,500000,'cartesian','fixed')

            avgAccuracy = avgAccuracy + balanced_accuracy_score(sampleLabelsIn, dataClusters)
            avgNumIter = avgNumIter + numIter
            
        avgAccuracy = avgAccuracy/100 
        avgNumIter = avgNumIter/100 

        #appends results
        results.append([r,num_points,avgAccuracy,avgNumIter])
        print(r,num_points,avgAccuracy,avgNumIter)


results = np.asarray(results)
print(results)

#stores results in a file
out_folder = os.path.join(cwd+ "/results/stereographic_sim_random_choice/classical")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

#saves results array in a file
string = file.partition(".")[0]
output_file = string + "_results_classical" + '_stereographic_random_choice'
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)


print('done')
