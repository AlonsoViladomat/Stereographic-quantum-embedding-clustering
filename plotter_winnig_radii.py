import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#detects current work directory
cwd = os.getcwd()

#loads files

# Files:  '2_7.txt' '6_6.txt' '8_6.txt' '10_7.txt'

error_parameter = '2_7.txt'

#creates plot folder to save
plot_folder = cwd + "\plots"
print(plot_folder)
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass

plot_folder = cwd + "\plots" + "\paper"
print(plot_folder)
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass


##### FOR 51K POINTS #####

"""
Loading Data
"""

file = "processed_result_overfitting_test_classical_" + error_parameter
file_2 = "accuracy_processed_result_timing_sim_stereo_classical_" + error_parameter
file_3 = "accuracy_processed_result_timing_sim_stereo_classical_" + error_parameter


in_folder = cwd+ "/Simulations/overfitting/classical_sim_random_choice"
in_folder_2 = cwd+ "/Simulations/overfitting/stereographic_sim_random_choice"
in_folder_3 = cwd+"/Simulations/overfitting/stereographic_sim_new_update"

results = np.loadtxt(os.path.join(in_folder, file))
results_2 = np.loadtxt(os.path.join(in_folder_2, file_2))
results_3 = np.loadtxt(os.path.join(in_folder_3, file_3))

#assigns values from result files
num_points = results_3[:,1]
num_points_no_repeat = results[:,0]
radii = results_3[:,0]
radii_no_repeat = radii[0:15]



accuracy_training = results[:,2] #training accuracies of plane 2d clustering 
accuracy_training_2 = results_2[:,3] #training accuracies of stereographic projection 3D clustering
accuracy_training_3 = results_3[:,3] #training accuracies of stereographic projection spherical clustering
errorbar_training = results[:,3]
errorbar_training_2 = results_2[:,4]
errorbar_training_3 = results_3[:,4]

accuracy_testing = results[:,4] #testing accuracies of plane 2d clustering 
accuracy_testing_2 = results_2[:,5] #testing accuracies of stereographic projection 3D clustering
accuracy_testing_3 = results_3[:,5] #testing accuracies of stereographic projection spherical clustering
errorbar_testing = results[:,5]
errorbar_testing_2 = results_2[:,6]
errorbar_testing_3 = results_3[:,6]

overfit = results[:,1] #mean(testting - training accuracy) of plane 2d clustering 
overfit_2 = results_2[:,2] #mean(testting - training accuracy) of stereographic projection 3D clustering
overfit_3 = results_3[:,2] #mean(testting - training accuracy) of stereographic projection spherical clustering

iterations = results[:,6]
iterations_2 = results_2[:,7]
iterations_3 = results_3[:,7]
iteration_deviation = results[:,7]
iteration_deviation_2 = results_2[:,8]
iteration_deviation_3 = results_3[:,8]

time = results[:,10] #training excecution time 2d clustering
time_2 = results_2[:,11] #training excecution time stereographic 3d clustering
time_3 = results_3[:,11] #training excecution time stereographic projection spherical clustering
time_deviation = results[:,11]
time_deviation_2 = results_2[:,12]
time_deviation_3 = results_3[:,12]

time_testing = results[:,12] #testing excecution time 2d clustering
time_testing_2 = results_2[:,13] #testing excecution time stereographic 3d clustering
time_testing_3 = results_3[:,13] #testing excecution time stereographic projection spherical clustering
time_deviation_testing = results[:,13]
time_deviation_testing_2 = results_2[:,14]
time_deviation_testing_3 = results_3[:,14]

#creates plot directory
plot_folder = cwd + "\plots" + "\paper" + "paper51K_comparison"
print(plot_folder)
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass

#clears previous plot figures and axes
plt.clf()
plt.cla()

"""
#### Produces multiple 2d plots for comparison (with errorbars) ####
"""
#### training accuracies ####

plt.errorbar(num_points_no_repeat,time, yerr=errorbar_training, capsize=3, capthick=1 ,label = "2D plane clustering")
plt.errorbar(num_points_no_repeat,time_2, yerr=errorbar_training_2, capsize=3, capthick=1 ,label="SP 3D clustering")
plt.errorbar(num_points_no_repeat,time_3, yerr=time_deviation_3, capsize=3, capthick=1 ,label="Quantum analogue clustering")


plt.plot(num_points_no_repeat,time, label="2D plane clustering", color = "yellow")
plt.plot(num_points_no_repeat,time_2, label="SP 3D clustering" , color = "blue")
plt.plot(num_points_no_repeat,time_3, label="Quantum analogue clustering", color = "red")

#sets legend and labels for the plot
x = "Number of points"
y = "Mean training accuracy"

plt.legend()

plt.xlabel(x)
plt.xscale('log')
plt.ylabel(y)

#saves plot
note = "Training_accuracy"
file_name = note + error_parameter[:-4] 
plt.savefig(os.path.join(plot_folder,file_name))

plt.grid()
plt.show()

plt.clf()
plt.cla()

### Plots the gains of the winning radii and annotates those radii ####
#####(training time)

#adds annotations to the plots
for x,y in zip(num_points_no_repeat,-time_2 + time):

    label = results_2[i,0]

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-5), # distance from text to points (x,y)
                 ha='left') # horizontal alignment can be left, right or center
    if x_change != x:
        x_change = x
        i = i+1

#adds annotations to the plots
for x,y in zip(np.log(num_points_no_repeat),-time_3 + time):

    label = results_3[i,0]

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-5), # distance from text to points (x,y)
                 ha='left') # horizontal alignment can be left, right or center
    if x_change != x:
        x_change = x
        i = i+1

#stupid dashed line at y = 0
plt.plot(num_points_no_repeat,np.zeros(len(num_points_no_repeat)),"k--") 

# produces plots together 
plt.plot(np.log(num_points_no_repeat),iterations, label = "2D plane clustering")
plt.plot(np.log(num_points_no_repeat),iterations_3, label="Quantum analogue clustering")

#sets legend and labels for the plot
x = "Number of points"
y = "Mean training iterations"

#title = "Mean training iterations of winner vs No. pts. ("+error_parameter[:-4].replace("_",".")+" dB)"
#plt.title(title)

plt.legend()

plt.xlabel(x)
plt.xscale('log')
plt.ylabel(y)

plt.grid()
plt.show()

#saves plot
note = "Training_iterations_gain"
file_name = note + x + " vs " + y + error_parameter[:-4] 
plt.savefig(os.path.join(plot_folder,file_name))

#clears plot figures and axes
plt.clf()
plt.cla()

### Plots the gains of the winning radii and annotates those radii ####
#####(iterations)

# Plots gains with reference 0 dashed line
plt.plot(num_points_no_repeat,np.zeros(len(num_points_no_repeat)),"k--") #stupid dashed line at y = 0

plt.plot(num_points_no_repeat,iterations - iterations_3,label="2D  clustering")
plt.plot(num_points_no_repeat,iterations - iterations_3,label="Quantum analogue clustering")

#adds annotations to the plots
i = 0
x_change = 0
for x,y in zip(num_points_no_repeat,iterations - iterations_3):

    label = results_3[i,0]

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='right') # horizontal alignment can be left, right or center
    
    if x_change != x:
        x_change = x
        i = i+1


#sets legend and labels for the plot
x = "Number of points"
y = "Mean training iteration gain"

#title = "Mean training iteration gain of winner vs No. pts. ("+error_parameter[:-4].replace("_",".")+" dB)"
#plt.title(title)

plt.legend()

plt.xlabel(x)
plt.ylabel(y)
plt.xscale('log')

#saves plot
note = "Training_iteration_gain"
file_name = x + " vs " + y + error_parameter[:-4] + note
plt.savefig(os.path.join(plot_folder,file_name))


##### FOR 260K POINTS #####

"""
Loading data
"""
"""
file = "processed_result_all_points_overfitting_test_classical_" + error_parameter
file_2 = "time_processed_result_timing_sim_stereo_classical_" + error_parameter
file_3 = "accuracy_processed_result_timing_sim_stereo_classical_" + error_parameter


in_folder = cwd+ "/Simulations/overfitting/all_points/classical_sim"
in_folder_2 = cwd+ "/Simulations/overfitting/stereographic_sim_random_choice"
in_folder_3 = cwd+"/Simulations/overfitting/all_points/stereo"

results = np.loadtxt(os.path.join(in_folder, file))
results_2 = np.loadtxt(os.path.join(in_folder_2, file_2))
results_3 = np.loadtxt(os.path.join(in_folder_3, file_3))

#assigns values from the loaded results
accuracy_train = results[:,2] #training accuracies of plane 2d clustering 
accuracy_3_train = results_3[:,3] #training accuracies of stereographic projection spherical clustering

accuracy_test = results[:,3] #testing accuracies of plane 2d clustering 
accuracy_3_test = results_3[:,5] #testing accuracies of stereographic projection spherical clustering

overfit = results[:,1] #mean(testting - training accuracy) of plane 2d clustering 
overfit_3 = results_3[:,2] #mean(testting - training accuracy) of stereographic projection spherical clustering

iterations = results[:,4]
iterations_3 = results_3[:,7]

time = results[:,6] #training excecution time 2d clustering
time_3 = results_3[:,11] #training excecution time stereographic projection spherical clustering

time = results[:,7] #testing excecution time 2d clustering
time_3 = results_3[:,13] #testing excecution time stereographic projection spherical clustering

#creates plot directory
plot_folder = cwd + "\plots" + "\paper" + "all_points_260K"
print(plot_folder)
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass

"""
### Plots for training accuracies
"""

# produces plots together 
plt.plot(np.log(num_points_no_repeat),accuracy_train, label = "2D plane clustering")
plt.plot(np.log(num_points_no_repeat),accuracy_3_train, label="Quantum analogue clustering")

#sets legend and labels for the plot
x = "Log(Number of points)"
y = "Mean training accuracy (%)"
title = "Mean training accuracy of winner vs No. pts. ("+error_parameter[:-4].replace("_",".")+" dB)"

plt.title(title)

plt.legend()

plt.xlabel(x)
plt.ylabel(y)

#saves plot
note = "Training_accuracy"
file_name = x + " vs " + y + error_parameter[:-4] + note
plt.savefig(os.path.join(plot_folder,file_name))

#clears plot figures and axes
plt.clf()
plt.cla()

# Plots gains with reference 0 dashed line
plt.plot(np.log(num_points_no_repeat),np.zeros(len(num_points_no_repeat)),"k--") #stupid dashed line at y = 0
plt.plot(np.log(num_points_no_repeat),accuracy_3_train - accuracy_train,label="Quantum analogue clustering")

#adds annotations to the plots
i = 0
x_change = 0
for x,y in zip(np.log(num_points_no_repeat),accuracy_3_train - accuracy_train):

    label = results_3[i,0]

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='right') # horizontal alignment can be left, right or center
    
    if x_change != x:
        x_change = x
        i = i+1


#sets legend and labels for the plot
x = "Log(Number of points)"
y = "Mean training accuracy gain (%)"
title = "Mean training accuracy gain of winner vs No. pts. ("+error_parameter[:-4].replace("_",".")+" dB)"

plt.title(title)

plt.legend()

plt.xlabel(x)
plt.ylabel(y)

#saves plot
note = "Training_accuracy_gain"
file_name = x + " vs " + y + error_parameter[:-4] + note
plt.savefig(os.path.join(plot_folder,file_name))

#clears plot figures and axes
plt.clf()
plt.cla()

"""
###Plots for testing accuracies
"""
# produces plots together
plt.plot(np.log(num_points_no_repeat),accuracy_test, label = "2D plane clustering")
plt.plot(np.log(num_points_no_repeat),accuracy_3_test, label="Quantum analogue clustering")

#saves plot
note = "Testing_accuracy"
file_name = x + " vs " + y + error_parameter[:-4] + note
plt.savefig(os.path.join(plot_folder,file_name))

#clears plot figures and axes
plt.clf()
plt.cla()

# Plots gains with reference 0 dashed line
plt.plot(np.log(num_points_no_repeat),np.zeros(len(num_points_no_repeat)),"k--") #stupid dashed line at y = 0
plt.plot(np.log(num_points_no_repeat),accuracy_3_test - accuracy_test,label="Quantum analogue clustering")

#adds annotations to the plots
i = 0
x_change = 0
for x,y in zip(np.log(num_points_no_repeat),accuracy_3_test - accuracy_test):

    label = results_3[i,0]

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='right') # horizontal alignment can be left, right or center
    
    if x_change != x:
        x_change = x
        i = i+1


#sets legend and labels for the plot
x = "Log(Number of points)"
y = "Mean testing accuracy gain (%)"
title = "Mean testing accuracy gain of winner vs No. pts. ("+error_parameter[:-4].replace("_",".")+" dB)"

plt.title(title)

plt.legend()

plt.xlabel(x)
plt.ylabel(y)

#saves plot
note = "Testing_accuracy_gain"
file_name = x + " vs " + y + error_parameter[:-4] + note
plt.savefig(os.path.join(plot_folder,file_name))

"""

"""
### Plots for training iterations
"""

# #shows the plot
plt.show()