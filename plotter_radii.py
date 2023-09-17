import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import LinearLocator
from matplotlib import cm

cwd = os.getcwd()
#mpl.use('pgf')
#plt.switch_backend

# Files:  '2_7.txt' '6_6.txt' '8_6.txt' '10_7.txt'

#reads results files
error_parameter = '2_7.txt'
file = "complete_processed_result_stereographic_overfitting_test_" + error_parameter
file_2 = "processed_result_overfitting_test_classical_" + error_parameter
#in_folder = cwd+"\Simulations\overfitting/all_points/new_update_stereo"
in_folder = cwd+"\Simulations\overfitting\stereographic_sim_new_update"
in_folder_2 = cwd+"/Simulations/overfitting/classical_sim_random_choice"

#loads data into arrays
results = np.loadtxt(os.path.join(in_folder, file))
results_2 = np.loadtxt(os.path.join(in_folder_2, file_2))

radii = results[:,0]
num_points = results[:,1]
accuracy = results_2[:,2]
accuracy_train = results[:,3]
accuracy_test = results[:,5]
standard_deviation_train = results[:,4]
standard_deviation_test = results[:,6]
iterations = results[:,7]
overfitting = results[:,2]
standard_deviation_iterations = results[:,8]
time = results[:,11]


x = radii
y = num_points
i = iterations
z = accuracy_train
w = accuracy_test
o = overfitting
t = time
st_dev = standard_deviation_iterations

"""
Only relevant for 54k points
"""
x = np.reshape(x, (8, 15))
y = np.reshape(y, (8, 15))
z = np.reshape(z, (8, 15))
w = np.reshape(w, (8, 15))
i = np.reshape(i, (8, 15))
o = np.reshape(o, (8, 15))
t = np.reshape(t, (8, 15))
st_dev = np.reshape(st_dev, (8, 15))

"""
"Only relevant for all points (240k)"
"""
"""
x = np.reshape(x, (10, 11))
y = np.reshape(y, (10, 11))
z = np.reshape(z, (10, 11))
"""

#We cut the data to only relevant radii
x_sliced = []
y_sliced = []
z_sliced = []
w_sliced = []
i_sliced = []
o_sliced = []
t_sliced = []

for i in range(8):      
    x_sliced.append(radii[4 + 15*i : 13 + 15*i])
    y_sliced.append(num_points[4 + 15*i : 13 + 15*i])
    z_sliced.append(accuracy_train[4 + 15*i : 13 + 15*i])
    w_sliced.append(accuracy_test[4 + 15*i : 13 + 15*i])
    i_sliced.append(iterations[4 + 15*i : 13 + 15*i])
    o_sliced.append(overfitting[4 + 15*i : 13 + 15*i])
    t_sliced.append(time[4 + 15*i : 13 + 15*i])

x_sliced = np.asarray(x_sliced)
y_sliced = np.asarray(y_sliced)
z_sliced = np.asarray(z_sliced)
w_sliced = np.asarray(w_sliced)
i_sliced = np.asarray(i_sliced)
o_sliced = np.asarray(o_sliced)
t_sliced = np.asarray(t_sliced)

"""
Surface Plot 
"""

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# surf = ax.plot_surface(x, y, t, cmap=cm.coolwarm)
surf = ax.plot_surface(x_sliced, y_sliced, w_sliced, cmap=cm.coolwarm)
ax.set_zlim(82.4, 86.9)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
#fig.colorbar(surf, shrink=0.5, aspect=5)


#ax.scatter(x, y, z, st_dev, marker = '|')
#ax.errorbar(x, y, z, st_dev)

plt.title("Mean testing accuracy vs radii vs number of points (" + error_parameter[:-4].replace("_",".")+" dB) \n" + "(Quantum analogue)")

ax.set_xlabel('log(radii)')
ax.set_ylabel('log(no. of points)')
ax.set_zlabel('accuracy (%)')

#plt.show()
"""

#plt.clf()
#plt.cla()


"""
Heatmap plots
"""

index = y_sliced[::-1]
index = np.around(np.log10(index[:,0]), decimals=2)
columns = x_sliced[0]
heatma = pd.DataFrame(w_sliced[::-1], index, columns)

sns.set(font_scale=1.8)
plt.subplots(figsize=(12,10))
sns.heatmap(heatma, annot=False)

plt.xlabel('Radii', fontsize = 22)
plt.ylabel(r'No. of points $\times 10$', fontsize = 22)
#plt.title('Quantum analogue training-testing accuracy heatmap (' + error_parameter[:-4].replace("_",".")+" dB)")
#plt.yscale('log')



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

plot_folder = cwd + "\plots" + "\paper" + "51K_comparison"
print(plot_folder)
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass

plot_folder = cwd + "\plots" + "\paper" + "51K_comparison" + "\Surface plots"
print(plot_folder)
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass

plot_folder = cwd + "\plots" + "\paper" + "51K_comparison" + "\Surface plots" + "\Heatmaps"
print(plot_folder)
try:
    os.mkdir(plot_folder)
except FileExistsError:
    pass

note = "spherical_"
file_name = 'testing_heatmap_of_' + note +error_parameter[:-4] + '_dB.png'
plt.savefig(os.path.join(plot_folder,file_name))
#plt.savefig(os.path.join(plot_folder,file_name), format = "pgf")

plt.show()