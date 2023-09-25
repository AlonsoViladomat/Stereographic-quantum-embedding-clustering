import psutil
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math as mt

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.providers.aer import QasmSimulator
from multiprocessing import Pool

num_cpus = psutil.cpu_count(logical=True)

def load_data(fileName):
    # Loads the data
    
    # INPUTS
    # fileName:str --- Name of the data file 
    
    # OUTPUTS
    # data: np.array --- Data
    
    with open(fileName, 'r') as file:
        data = file.read()      
        data = data.split("\n")
        data = [item.split(",") for item in data]
        data = data[:-1]
        data = [[float(item) for item in sublist] for sublist in data]
        data = np.asarray(data)
    return data

def generate_data(r, n_1, n_2, k, N):
    # Generates QAM data

    # INPUTS
    # r: float --- Minimum noiseless radius of the QAM constellation
    # n_1: float --- Variance of the Amplitude Noise
    # n_2: float --- Variance of the Angular noise
    # k : int --- k of k-mean. Decides the initial number of centroids.
    # N : int --- Number of data points

    # OUTPUTS
    # noisyData: np.array --- Noise added k-QAM data

    labels = np.random.randint(1, k + 1, [N, 1])

    noise_x = np.random.normal(0, n_1, [N, 1])
    noise_y = np.random.normal(0, n_1, [N, 1])
    angularNoise = np.random.normal(0, n_2, [N, 1])

    theta = (((labels - 1) // 4) % 2) * np.pi / 4 + (labels % 4 - 1) * np.pi / 2
    amplitude = 1 + (labels - 1) // 4

    data = np.concatenate(
        (r *
         amplitude *
         np.cos(theta),
         r *
         amplitude *
         np.sin(theta),
         labels),
        axis=1)
    noisyData = np.concatenate(
        (r *
         amplitude *
         np.cos(
             theta +
             angularNoise) +
            noise_x,
            r *
            amplitude *
            np.sin(
             theta +
             angularNoise) +
            noise_y,
            labels),
        axis=1)

    return noisyData


def polar_to_cartesian(data):
    # transforms polar coordinates to cartesian

    # INPUTS
    # data:np.array --- Polar Data

    # OUTPUTS
    # data: np.array --- Cartesian Data

    cartesianData = np.concatenate(
        (data[
            :,
            [0]] *
            np.cos(
            data[
                :,
                [1]]),
            data[
            :,
            [0]] *
            np.cos(
            data[
                :,
                [1]])),
        axis=1)

    return cartesianData


def cartesian_to_polar(data):
    # transforms cartesian coordinates to polar

    # INPUTS
    # data:np.array --- Cartesian Data

    # OUTPUTS
    # data: np.array --- Polar Data

    r = np.linalg.norm(data, axis=1, keepdims=True)
    theta = np.arctan2(data[:, [1]], data[:, [0]]) % (2 * np.pi)
    polarData = np.concatenate((r, theta), axis=1)

    return polarData


def rotate(data, angle):
    # Rotates the data for the given angle

    # INPUTS
    # data: np.array --- 2d Data that is rotated
    # angle: float --- Angle of the Rotation in radian

    # OUTPUTS
    # rotatedData: np.array   --- Rotated data

    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).T
    rotatedData = np.dot(data, rotation)

    return rotatedData


def classical_to_quantum(vector, cname):
    # Returns a quantum circuitry that constructs the quantum state of the input vector

    # INPUTS
    # vector: np.array or list --- 2d Vector that will be constructed
    # cname: str --- Name of the circuit

    # OUTPUTS
    # circ: QuantumCircuit  ---

    circ = QuantumCircuit(1, name=cname)
    circ.u(vector[0], vector[1], 0, 0)
    return circ


def swap_test(vector1, vector2):
    # Returns the circuitry of swap tests for the given 2d vectors

    # INPUTS:
    # vector1,vector2: np.array or list ---  2d Inputs of the swap test

    # OUTPUTS
    # circ3: QuantumCircuit --- Swap test quantum circuitry

    circ1 = classical_to_quantum(vector1, "subcirc1")
    circ2 = classical_to_quantum(vector2, "subcirc2")
    circ = QuantumCircuit(3, 1)
    circ.h(0)
    circ.append(circ1, [1])
    circ.append(circ2, [2])
    circ.cswap(0, 1, 2)
    circ.h(0)
    circ.measure(0, 0)
    return circ


def parallel_test(circs):
    # Constructs a parallel Swap test Circuitry from the list of input swap tests

    # INPUTS:
    # circs: list(QuantumCircuit) ---  List of Swap test Quantum circuitry

    # OUTPUTS
    # circ3: QuantumCircuit --- Parallel Swap test quantum circuitry

    circLen = len(circs)
    parallelCirc = QuantumCircuit(3 * circLen, circLen)

    for i in range(circLen):
        parallelCirc.append(circs[i], [3 * i, 3 * i + 1, 3 * i + 2], [i])
    return parallelCirc


def random_seeds(k, data):
    # Returns list of k different integers from 0 to len(data)-1

    # INPUTS:
    # k: int ---

    # OUTPUTS:
    # data: np.array ---

    seedNums = np.random.choice(range(data.shape[0]), k, replace=False)
    return seedNums


def circuit_simulator(circ,backend):
    # Simulates a given Quantum Circuit
    
    # INPUTS:
    # circ: QuantumCircuit --- Quantum circuitry 
    # backend: qiskit.backends --- Backend
    
    # OUTPUTS:
    # counts: dict --- Result of Simulation
    
    qc_compiled = transpile(circ, backend)
    job_sim = backend.run(qc_compiled, shots=2048)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc_compiled)
    return counts


def embed_data(data, coordinateSystem, rMax, rMin):
    # Transforms a given 2d data according to angle encoding.

    # INPUTS:
    # data: np.array ---
    # coordinateSystem: string --- 'polar' or 'cartesian', indicates the coordinate system used
    # rMax: Maximum length in the whole QAM data
    # rMin: Minimum length in the whole QAM data

    # OUTPUTS:
    # counts: np.array  --- Angle encoded Data

    #  Finding the maximum length in the data and implementing normalized angle embedding.

    # Cartesian System:
    # [x1 x2] -> pi*( [x1+r x2+r]/(2*r)) Coded s.t [theta, phi] is between [0,0] to [pi,pi]

    # Polar System:
    # [r angle] -> [pi/12 +(5pi/6)*(r-rmin)/(rmax-rmin) , angle] Coded s.t [theta, phi] is between [pi/12,11pi/12] to [0,2pi]

    if coordinateSystem == 'polar':
        transformedData = cartesian_to_polar(data)
        transformedData[:, 0] = (np.pi / 12) + (5 * np.pi / 6) * \
            (transformedData[:, 0] - rMin) / (rMax - rMin)
        transformedData[:, 1] = transformedData[:, 1]

    else:
        transformedData = np.pi * (np.ones(data.shape) * rMax + data) / (2 * rMax)


    return transformedData


def sample_data(k, data, labels):
    # Samples k number of points from the data.

    # INPUTS:
    # k:int --- Number of samples
    # data: np.array ---
    # labels: np.array ---

    # OUTPUTS:
    # sampleData: np.array  ---

    randomNums = np.random.choice(range(data.shape[0]), k, replace=False)
    sampleData = data[randomNums, :]
    sampleLabels = labels[randomNums]
    return [sampleData, sampleLabels]


def initialize_centroids(k,data,method,centroids):
    # Initializes Centroids for kmean algorithm for the given method. 
    
    # INPUTS:
    # k:int ---
    # data: np.array ---
    # method: str --- Method for assigning initial centroids 'random' or 'maxEstimate'
    # centroids: np.array ---
    
    # OUTPUTS:
    # sampleData: np.array  ---
    
    # If method is 'random' centroids are chosen randomly from the data
    if method == "random":
        seeds = random_seeds(k, data)
        centroids[:,[0,1]] = data[seeds]
        
    # If method is 'maxEstimate' maximum vector length of the data is found. Centroids are chosen according to QAM constellation    
    
    if method == "maxEstimate":
        r = np.amax(np.linalg.norm(data,axis = 1))/4
        centroids[:,[0,1]] = np.array([[r*i*np.cos((j + (1-i%2)/2 )*np.pi/2),r*i*np.sin((j + (1-i%2)/2 )*np.pi/2)]
        for i in range(1,2 + k//4) for j in range(4) if 4*(i-1)+j < k])
    
    else:
        centroids[:,[0,1]] = np.array([[x,y] for x in np.linspace(-1.1,1.1,8) for y in np.linspace(-1.1,1.1,8)])

    return centroids


def qk_mean(k, data, fixed_centroids,maxNumIter, numParallelTests, coordinateSystem, method='fixed'):
    # Quantum K-mean Algorithm

    # INPUTS:
    # k: int --- k of k-mean. Decides the initial number of centroids.
    # data: np.array ---
    # fixed_centroids --- this array sets initial given points as centroids (only usable for method = "fixed")
    # maxNumIter: int --- Maximum number of iteration
    # numParallelTests: int --- Number of Swap tests parallelized and simulated at once.
    # coordinateSystem: str --- 'polar' or 'cartesian', indicates the coordinate system used
    # method: str --- Method for assigning initial centroids 'random' or 'maxEstimate'

    # OUTPUTS:
    # dataClusters: np.array --- Output Clusters of the algorithm

    backend = QasmSimulator()

    # Clusters Before is used for understanding the changes between iterations
    dataClusters = np.zeros(len(data))
    clustersBefore = np.zeros(len(data))

    centroids = np.zeros([k,3])
    centroids[:,2] = np.arange(1, k+1)

    # Initializing Random Centroids from data
    if method == "fixed":
        centroids[:, [0, 1]] = fixed_centroids
    else:
        # Initializing Random Centroids from data
        centroids = initialize_centroids(k,data,method,centroids) 

    # Angle Embedded Centroids
    transformedCentroids = np.zeros([k, 3])
    transformedCentroids[:, 2] = np.arange(1, k + 1)

    # Embedding data
    rMax = np.amax(np.linalg.norm(data, axis=1))
    rMin = np.amin(np.linalg.norm(data, axis=1))
    transformedData = embed_data(data, coordinateSystem, rMax, rMin)

    transformedCentroids[:, [0, 1]] = embed_data(centroids[:, [0, 1]], coordinateSystem, rMax, rMin)

    didClustersChange = False
    numIter = 0
    while not didClustersChange and numIter < maxNumIter:

        np.copyto(clustersBefore, dataClusters)

        # Assigning data points to different clusters
        dataClusters = cluster_assignment(
            transformedData,
            transformedCentroids,
            dataClusters,
            backend,
            numParallelTests,
            numIter)

        if np.array_equal(dataClusters, clustersBefore, equal_nan=False):
            didClustersChange = True

        # Deleting empty cluster centroids
        delList = []
        for i in np.arange(len(centroids)):
            if np.sum([dataClusters == centroids[i, 2]]) == 0:
                delList.append(i)
        centroids = np.delete(centroids, delList, 0)
        transformedCentroids = np.delete(transformedCentroids, delList, 0)

        # Finding centroids from data by taking the mean of points in the clusters.

        if coordinateSystem == 'cartesian':
            for i in np.arange(len(centroids)):
                centroids[i, [0, 1]] = data[dataClusters == centroids[i, 2]].mean(axis=0)
        if coordinateSystem == 'polar':
            for i in np.arange(len(centroids)):
                x = np.sum(data[dataClusters == centroids[i, 2]], axis=0)
                theta = np.arctan2(x[1], x[0]) % (2 * np.pi)
                r = np.mean(np.linalg.norm(data[dataClusters == centroids[i, 2]], axis=1))
                centroids[i, [0, 1]] = np.array([r * np.cos(theta), r * np.sin(theta)])

        transformedCentroids[:, [0, 1]] = embed_data(
            centroids[:, [0, 1]], coordinateSystem, rMax, rMin)

        numIter = numIter + 1       


    return dataClusters, centroids, numIter

def cluster_assignment(data,centroids,dataClusters,backend,numParallelTests,numIter):
    # Assings clusters to each point in data
    
    # INPUTS:
    # data: np.array --- 
    # centroids: np.array(k,3) --- 
    # dataClusters: np.array(k,1) --- 
    # backend: qiskit.backends --- Backend
    # numParallelTests: int --- Number of Swap tests parallelized and simulated at once.
    # numIter : int --- Number of iteration algorithm passed
    
    # OUTPUTS:
    # dataClusters: np.array --- Output Clusters after iteration
    
    for point in range(0,len(data),numParallelTests): 
        
        # Number of Swap Tests done in parallel.
        numTests = min(numParallelTests,len(data)-point) #for leftover data (when datapoints left < no. of available || swap tests)
        
        #  Creating parallel circuitry for numTest points for a given centroid. Doing this for every centroid.
        circs = [[swap_test(data[point+i,:],centroids[centroid,[0,1]]) for i in range(numTests)] for centroid in range(len(centroids))]
        parallelCircs = [parallel_test(circ) for circ in circs]
        
        """
        if point == 0 and numIter == 0:
            plt.figure(parallelCircs[0].decompose().draw(output='mpl', style={'backgroundcolor': '#EEEEEE'}))
            plt.show()
        """
        
        # List of Results from parallel circuitry. Results are in the form of dict, Dict values being numTests bits.
        results = [circuit_simulator(parallelCirc,backend) for parallelCirc in parallelCircs] #Is the number of parallel circuits the number of shots?wtf?
        
        # Finding the Clusters from the data.
        for k in range(numTests):
            dataClusters[point + k] = centroids[np.argmax([sum([result[i] for i in result if i[numTests-k-1]== '0']) for result in results]),2]
            
    return dataClusters


def initialize_plot():
    # Initializes the plot outputs figure and axis of the initialized plot.

    # OUTPUTS:
    # fig: matplotlib.pyplot.figure ---
    # ax: matplotlib.pyplot.axis ---

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.cla()
    plt.title('fakeData', fontsize=10)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('y', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    return [fig, ax]


def plot_dataClusters(dataClusters, fig, ax, numIter, centroids, data):
    # Plots the data

    # INPUTS:
    # data: np.array ---
    # centroids: np.array(k,3) ---
    # dataClusters: np.array(k,1) ---
    # fig: matplotlib.pyplot.figure ---
    # ax: matplotlib.pyplot.axis ---

    ax.clear()
    for i in centroids[:, 2]:
        ax.plot(data[dataClusters == i, 0], data[dataClusters == i, 1], '.', label=i,
                markersize=4, color=[i / 64, 1 / i, ((64 * (i % 2)) + i * pow(-1, (i % 2))) / 64])
    for i in range(len(centroids)):
        j = centroids[i, 2]
        ax.plot(centroids[i, 0], centroids[i, 1], '.', markersize=8, color=[
                j / 64, 1 / j, ((64 * (j % 2)) + j * pow(-1, (j % 2))) / 64])
    #ax.legend(loc='upper left')
    ax.set_title(numIter)
    #plt.show()
    # fig.canvas.draw()


def classical_k_mean(k,data,fixed_centroids,maxNumIter,coordinateSystem,method = 'maxEstimate'):
    # Classical K-mean Algorithm
    
    # INPUTS:
    # k: int --- k of k-mean. Decides the initial number of centroids.
    # data: np.array --- 
    # fixed_centroids: np.array --- this array sets initial points as centroids (only usable for method = "fixed")
    # coordinateSystem: str --- 'polar' or 'cartesian', indicates the coordinate system used
    # maxNumIter: int --- Maximum number of iteration
    # method: str --- Method for assigning initial centroids 'random' or 'maxEstimate'

    
    # OUTPUTS:
    # dataClusters: np.array --- Output Clusters of the algorithm 
    
    # Clusters Before is used for understanding the changes between iterations
    dataClusters = np.zeros(len(data))
    clustersBefore = np.zeros(len(data))
    
    # For a point in centroids [x1,x2,x3] x1 and x2 are coordinates. x3 is the Centroid number ranging from 1 to
    centroids = np.zeros([k,3])
    centroids[:,2] = np.arange(1, k+1)

    if method == "fixed":
        centroids[:, [0, 1]] = fixed_centroids
    else:
        # Initializing Random Centroids from data
        centroids = initialize_centroids(k,data,method,centroids) 

    """print(centroids)"""
    
    # Plotting the data Clusters
    #[fig,ax] = initialize_plot()
    #plot_dataClusters(dataClusters,fig,ax, -1 ,centroids,data)
    
    didClustersChange = False
    numIter = 0
    
    while not didClustersChange and numIter < maxNumIter:
        
        np.copyto(clustersBefore,dataClusters)
        
        # Assigning data points to different clusters according to nearest centroid.
        dataClusters = centroids[np.argmin(np.linalg.norm(np.tile(data,(len(centroids),1,1)) - np.reshape(centroids[:,[0,1]],(len(centroids),1,2)),axis = 2),axis = 0),2]
        
        # Uses normalized version of the polar data and assigns data points to nearest cenroids

        
        if np.array_equal(dataClusters, clustersBefore, equal_nan=False):
            didClustersChange = True
        
        # Deleting empty cluster centroids
        delList = []
        for i in np.arange(len(centroids)): 
            if np.sum([dataClusters == centroids[i,2]]) == 0:
                delList.append(i)
        centroids = np.delete(centroids, delList, 0)
        
        # Plotting the data Clusters
       # [fig,ax] = initialize_plot()
        #plot_dataClusters(dataClusters,fig,ax,numIter,centroids,data)
        
        # Finding centroids from data by taking the mean of points in the clusters
        if coordinateSystem == 'cartesian':
            for i in np.arange(len(centroids)):
                centroids[i,[0,1]] = data[dataClusters == centroids[i,2]].mean(axis = 0)
        
        if coordinateSystem == 'polar':
            for i in np.arange(len(centroids)):
                x = np.sum(data[dataClusters == centroids[i,2]],axis = 0)
                theta = np.arctan2(x[1],x[0])% (2*np.pi)
                r = np.mean(np.linalg.norm(data[dataClusters == centroids[i,2]],axis = 1))
                centroids[i,[0,1]] = np.array([r*np.cos(theta),r*np.sin(theta)])
                
        
        numIter = numIter + 1


    return dataClusters, centroids, numIter

 
# This function generates all n bit Gray
# codes and prints the generated codes
def generateGrayarr(n):
 
 
    # This code is contributed
    # by Mohit kumar 29
    
    # base case
    if (n <= 0):
        return
 
    # 'arr' will store all generated codes
    arr = list()
 
    # start with one-bit pattern
    arr.append("0")
    arr.append("1")
 
    # Every iteration of this loop generates
    # 2*i codes from previously generated i codes.
    i = 2
    j = 0
    while(True):
 
        if i >= 1 << n:
            break
     
        # Enter the previously generated codes
        # again in arr[] in reverse order.
        # Nor arr[] has double number of codes.
        for j in range(i - 1, -1, -1):
            arr.append(arr[j])
 
        # append 0 to the first half
        for j in range(i):
            arr[j] = "0" + arr[j]
 
        # append 1 to the second half
        for j in range(i, 2 * i):
            arr[j] = "1" + arr[j]
        i = i << 1
    
    return arr

def bit_to_decimal(data):
    #takes an array of bits and maps it to an array of decimals

    data_aux = []
    for x in data:
        aux = 0
        for k in range(len(x)):
            aux= aux + x[k]*2**((len(x)-1)-k)
    
        data_aux.append(aux)
    return data_aux

def bitstrings_to_decimal(data):
    #takes a bit array in string format and maps it to its decimal
    arr_aux = []
    for i in range(len(data)):
        arr_aux.append(int(data[i],2))
    return arr_aux


#This class creates a two way map relating a-> b and b->a
class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2

