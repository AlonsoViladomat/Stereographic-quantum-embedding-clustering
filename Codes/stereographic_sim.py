import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from functions_3D import *
from mpl_toolkits.mplot3d import Axes3D

# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'
file_name = os.path.join(cwd+"/realdata", file)
file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

num_point = [320,640,1280,2560,3200,6400,12800,25600,52124]
radii = [0.01,0.1,0.25,0.5,1,1.5,2,5,10,100]

#num_points = 52124
#r = 2
results = []
alphabet = scipy.io.loadmat(file_name_2)
alphabet = alphabet['alphabet']

for num_points in num_point:
    for r in radii:

        data = scipy.io.loadmat(file_name)

        rxsignal = data['rxsignal']

        rxsignal = rxsignal[0:num_points]
        rxsignal = rxsignal[:,0]

        initial_bits = scipy.io.loadmat(file_name)["bits"]
        initial_bits = bit_to_decimal(initial_bits)
        sampleLabels = np.add(initial_bits,1)
        sampleLabels = sampleLabels[0:num_points]

        d_x = (r*r)*2*rxsignal.real/(r*r + np.absolute(rxsignal)*np.absolute(rxsignal))
        d_y = (r*r)*2*rxsignal.imag/(r*r + np.absolute(rxsignal)*np.absolute(rxsignal))
        d_z = r*(-r*r + np.absolute(rxsignal)*np.absolute(rxsignal))/(r*r + np.absolute(rxsignal)*np.absolute(rxsignal))

        #Alphabet Tx for initial centroid
        alph_x = (r*r)*2*alphabet.real/(r*r + np.absolute(alphabet)**2)
        alph_y = (r*r)*2*alphabet.imag/(r*r + np.absolute(alphabet)**2)
        alph_z = r*(-r*r + np.absolute(alphabet)**2)/(r*r + np.absolute(alphabet)**2)

        tx_init_pt = []
        for i in range(len(rxsignal)):
                tx_init_pt.append([d_x[i],d_y[i],d_z[i]])
        tx_init_pt = np.asarray(tx_init_pt)

        tx_alph = []
        for i in range(len(alphabet)):
                tx_alph.append([float(alph_x[i]),float(alph_y[i]),float(alph_z[i])])
        tx_alph = np.asarray(tx_alph)


        dataClusters, centroids, numIter = classical_k_mean(64,tx_init_pt,tx_alph,500000,'cartesian','fixed')

        #Creates directory to store plots
        plot_folder = os.path.join(cwd+"/plots/stereographic_sim/classical")
        try:
            os.mkdir(plot_folder)
        except FileExistsError:
            pass

        # Plotting the data Clusters
        string = file.partition("=")[2]
        [fig,ax] = initialize_plot()
        plot_dataClusters(dataClusters,fig,ax,numIter,centroids,tx_init_pt)
        plt.title("Classical k-means (stereographic)" +" P=" +string.partition(".")[0])

        print(string)
        #plt.show()

        #Saves plot file
        string = file.partition(".")[0]
        plot_file_name = "Classical_k_means" + string + 'stereographic_r='+ str(r).replace(".",",") + "num_points=" + str(num_points)
        plt.savefig(os.path.join(plot_folder,plot_file_name))


        accuracy = balanced_accuracy_score(sampleLabels, dataClusters)
        cfMatrix = confusion_matrix(sampleLabels, dataClusters)

        #Creates directory and stores the accuracy confusion matrix plot
        plot_folder = os.path.join(cwd+ "/plots/stereographic_sim/classical/accuracy")
        try:
            os.mkdir(plot_folder)
        except FileExistsError:
            pass

        fig, ax = plt.subplots(figsize=(10,10)) 
        sns.heatmap(cfMatrix,ax = ax, fmt='g')
        string = file.partition("=")[2]
        print(string)
        plt.title("Confusion matrix of classical k-means (stereographic r="+ str(r) + ")"+" P=" +string.partition(".")[0] + " accuracy=" + str(np.round(accuracy,4)))

        #saves confusion matrix file to directory
        string = file.partition(".")[0]
        plot_file_name = string+"_confusion_matrix" + "classical" + '_stereographic_r=' + str(r).replace(".",",") + "num_points=" + str(num_points)
        print(plot_file_name)
        plt.savefig(os.path.join(plot_folder,plot_file_name))

        #appends results
        results.append([r,num_points,accuracy])
        print(r,num_points,accuracy)


results = np.asarray(results)
print(results)

#stores results in a file
out_folder = os.path.join(cwd+ "/results/stereographic_sim/classical")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

#saves results array in a file
string = file.partition(".")[0]
output_file = string + "_results_classical" + '_stereographic'
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)


print('done')
