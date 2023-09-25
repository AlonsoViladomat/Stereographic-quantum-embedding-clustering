import os

import scipy.io
import json

from functions import *

cwd = os.getcwd() #current working directory

file_name = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

data = scipy.io.loadmat(file_name)

file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

#fixing intitial centroids for k-means as the initial points of transmission
fixed_centroids = scipy.io.loadmat(file_name_2)["alphabet"]
fixed_centroids = np.reshape(fixed_centroids,fixed_centroids.size)
fixed_centroids = np.column_stack([fixed_centroids.real, fixed_centroids.imag])

#readind the real-world data
data = data["rxsignal"]
data = np.reshape(data,data.size)
data = np.column_stack([data.real, data.imag])

#reading the initial transmission labels
initial_bits = scipy.io.loadmat(file_name)["bits"]
initial_bits = bit_to_decimal(initial_bits)
sampleLabels = np.add(initial_bits,1)
            
#print(data)
#print(fixed_centroids)
numIters = [5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000]
shots = [5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000]
numPoints = [320,640,1280,2560,3200,6400,12800,25600,52124]
for numPoint in numPoints
    for maxNumIter in numIters
        for shot in shots
            
            #taking different numbers of datapoints
            dataIn = data[0:numPoint]
            dataIn = dataIn[:,0]
            
            sampleLabels = sampleLabels[0:numPoint]
                        
            
            
            #MAIN FUNCTION CALL
            dataClusters, centroids, numIter = qk_mean(64 , dataIn , fixed_centroids, maxNumIter , shot , "cartesian" , "fixed")
            
            
            #Creates directory to store plots
            plot_folder = os.path.join(cwd+"/plots/qkmeans_rmax_version/cluster_plots")
            try:
                os.mkdir(plot_folder)
            except FileExistsError:
                pass

            # Plotting the data Clusters
            string = file.partition("=")[2]
            [fig,ax] = initialize_plot()
            plot_dataClusters(dataClusters,fig,ax,numIter,centroids,fixed_centroids)
            plt.title("Quantum k-means rmax version " +" P=" +string.partition(".")[0])

            print(string)
            #plt.show()

            #Saves plot file
            string = file.partition(".")[0]
            plot_file_name = "Quantum_k_means" + string + 'max_num_iter=' + str(maxNumIter) + "num_points=" + str(numPoint) + "shots=" + str(shot)
            plt.savefig(os.path.join(plot_folder,plot_file_name))


            accuracy = balanced_accuracy_score(sampleLabels, dataClusters)
            cfMatrix = confusion_matrix(sampleLabels, dataClusters)

            #Creates directory and stores the accuracy confusion matrix plot
            plot_folder = os.path.join(cwd+ "/plots/qkmeans_rmax_version/accuracy_cf")
            try:
                os.mkdir(plot_folder)
            except FileExistsError:
                pass

            fig, ax = plt.subplots(figsize=(10,10)) 
            sns.heatmap(cfMatrix,ax = ax, fmt='g')
            string = file.partition("=")[2]
            print(string)
            plt.title("Confusion matrix of quantum k-means" + " P=" +string.partition(".")[0] + " accuracy=" + str(np.round(accuracy,4)))

            #saves confusion matrix file to directory
            string = file.partition(".")[0]
            plot_file_name = string+"_confusion_matrix" + "quantum" + 'max_num_iter=' + str(maxNumIter) + "num_points=" + str(numPoint) + "shots=" + str(shot)
            
            #print(plot_file_name)
            plt.savefig(os.path.join(plot_folder,plot_file_name))

            #appends results
            results.append([numPoint, maxNumIter, shot, numIter, accuracy])
            print(numPoint, maxNumIter, shot, numIter, accuracy)


results = np.asarray(results)
print(results)

#stores results in a file
out_folder = os.path.join(cwd+ "/results/qkmeans_rmax_version")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

#saves results array in a file
string = file.partition(".")[0]
output_file = string + "_results_qkmeans" + '_rmax_version'
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)

print('done')
        