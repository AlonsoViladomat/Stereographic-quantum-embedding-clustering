import os
import time
import scipy.io

from functions import *

accuracyScores = list()

# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

#establishing file names and current work directory
cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat'
file_name = os.path.join(cwd+"/realdata", file)
file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

#loading and assigning data to arrays
data = scipy.io.loadmat(file_name)["rxsignal"]
#data = data[0:320] #this is just in case one wants to truncate data
column = 0 #choses which device from the detected real data
data = data[:,column]
data = data.reshape(data.size)
data = np.column_stack([data.real, data.imag])
#print(data,data.size)

#loads fixed centroids
fixed_centroids = scipy.io.loadmat(file_name_2)["alphabet"]
fixed_centroids = np.reshape(fixed_centroids,fixed_centroids.size)
fixed_centroids = np.column_stack([fixed_centroids.real, fixed_centroids.imag])

#loads initial expected bits to each collected point in data
initial_bits = scipy.io.loadmat(file_name)["bits"]
initial_bits = bit_to_decimal(initial_bits)
#initial_bits = initial_bits[0:320] #again, just in case one decided to truncate data points

#start counting excecution time
start_time = time.time()

#excecutes quantum k means algorithm
dataClusters, centroids, numIter = classical_k_mean(64, data, fixed_centroids, 500000, 'cartesian', 'random')
sampleLabels = np.add(initial_bits,1)

#counts time passed after the excecution of the quantum k-means algorithm
end_time = time.time()

#Creates directory to store plots
plot_folder = os.path.join(cwd+"/plots/classical")
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass

# Plotting the data Clusters
[fig,ax] = initialize_plot()
plot_dataClusters(dataClusters,fig,ax,numIter,centroids,data)
string = file.partition("=")[2]
print(string)
plt.title("Classical k-means algorithm" +" P=" +string.partition(".")[0])
#plt.show()

#Saves plot file
string = file.partition(".")[0]
plot_file_name = "Classical_k_means" + string + '_maxEstimate'
plt.savefig(os.path.join(plot_folder,plot_file_name))

#calculates the confusion matrix and the accuracy between expected clusters with predicted ones
accuracy = balanced_accuracy_score(sampleLabels, dataClusters)
cfMatrix = confusion_matrix(sampleLabels, dataClusters)

#Creates directory and stores the accuracy confusion matrix plot
plot_folder = os.path.join(cwd+ "/plots/accuracy")
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass

fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(cfMatrix,ax = ax, fmt='g')
string = file.partition("=")[2]
print(string)
plt.title("Confusion matrix of classical algorithm"+ " (" + str(column) + ")" +" P=" +string.partition(".")[0] + " accuracy=" + str(np.round(accuracy,4)))
#plt.show()

#saves confusion matrix file to directory
string = file.partition(".")[0]
plot_file_name = string+"_confusion_matrix" + "_(" + str(column) + ")_" + "classical" + '_maxEstimate'
print(plot_file_name)
plt.savefig(os.path.join(plot_folder,plot_file_name))

#displays and stores results
results = np.asarray([len(data),accuracy,end_time-start_time])
print(results)

out_folder = os.path.join(cwd+ "/results")
try:
    os.mkdir(out_folder)
except FileExistsError:
    pass

string = file.partition(".")[0]
output_file = string + "_results_classical" + '_maxEstimate'
np.save(os.path.join(out_folder , output_file), results)
np.savetxt(os.path.join(out_folder , output_file), results)

print("done!")

