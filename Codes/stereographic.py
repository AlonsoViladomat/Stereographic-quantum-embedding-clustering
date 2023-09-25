import os
import numpy
import scipy.io
import matplotlib.pyplot as plotlib

from functions_3D import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans as km

# Files:  'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=6_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=8_6dBm_80kmSSMF.mat' 'DP_64QAM_80_GBd_P=10_7dBm_80kmSSMF.mat'

cwd = os.getcwd() 
file = 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat'
file_name = os.path.join(cwd+"/realdata", file)
file_name_2 = os.path.join(cwd+"/realdata", 'DP_64QAM_80_GBd_P=2_7dBm_80kmSSMF.mat')

data = scipy.io.loadmat(file_name)

alphabet = data['alphabet']
rxsignal = data['rxsignal']

#rxsignal = rxsignal[0:5000]
rxsignal = rxsignal[:,0]

initial_bits = scipy.io.loadmat(file_name)["bits"]
initial_bits = bit_to_decimal(initial_bits)
sampleLabels = np.add(initial_bits,1)
#sampleLabels = sampleLabels[0:5000]

"""
# Limit the amount of data
number_of_rows = 500
choice_of_rows = numpy.random.choice(range(rxsignal.shape[0]), size=number_of_rows, replace=False)
rxsignal = rxsignal[choice_of_rows,:]
"""

# Choose the radius of the sphere
r = 1

d_x = (r*r)*2*rxsignal.real/(r*r + numpy.absolute(rxsignal)*numpy.absolute(rxsignal))
d_y = (r*r)*2*rxsignal.imag/(r*r + numpy.absolute(rxsignal)*numpy.absolute(rxsignal))
d_z = r*(-r*r + numpy.absolute(rxsignal)*numpy.absolute(rxsignal))/(r*r + numpy.absolute(rxsignal)*numpy.absolute(rxsignal))

#Alphabet Tx for initial centroid
alph_x = (r*r)*2*alphabet.real/(r*r + numpy.absolute(alphabet)**2)
alph_y = (r*r)*2*alphabet.imag/(r*r + numpy.absolute(alphabet)**2)
alph_z = r*(-r*r + numpy.absolute(alphabet)**2)/(r*r + numpy.absolute(alphabet)**2)

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
plot_folder = os.path.join(cwd+"/plots/classical")
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
plot_file_name = "Classical_k_means" + string + 'stereographic_r='+ str(r).replace(".",",")
plt.savefig(os.path.join(plot_folder,plot_file_name))


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
plt.title("Confusion matrix of classical k-means (stereographic r="+ str(r) + ")"+" P=" +string.partition(".")[0] + " accuracy=" + str(np.round(accuracy,4)))
print(str(np.round(accuracy,4)))

#saves confusion matrix file to directory
string = file.partition(".")[0]
plot_file_name = string+"_confusion_matrix" + "classical" + '_stereographic_r=' + str(r).replace(".",",")
print(plot_file_name)
plt.savefig(os.path.join(plot_folder,plot_file_name))


#plt.show()
print('done')
