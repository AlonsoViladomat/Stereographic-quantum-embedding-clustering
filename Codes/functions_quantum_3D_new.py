import psutil
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math as mt

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from qiskit import QuantumCircuit
from qiskit import Aer, transpile, IBMQ
from qiskit.providers.aer import QasmSimulator
from multiprocessing import Pool

num_cpus = psutil.cpu_count(logical=True)


def classical_to_quantum(vector, cname):
    # Returns a quantum circuitry that constructs the quantum state of the input vector using angle embedding

    # INPUTS
    # vector: np.array or list --- 2d Vector that will be constructed
    # cname: str --- Name of the circuit

    # OUTPUTS
    # circ: QuantumCircuit  ---

    circ = QuantumCircuit(1, name=cname)
    circ.u(vector[0], vector[1], 0, 0)
    return circ


def bell_test(vector1, vector2):
    # Returns the circuitry of swap tests for the given 2d vectors

    # INPUTS:
    # vector1,vector2: np.array or list ---  2d Inputs of the swap test

    # OUTPUTS
    # circ3: QuantumCircuit --- Swap test quantum circuitry
    #print("entered bell test") # debug

    circ1 = classical_to_quantum(vector1, "UGate_Angle_Embedding_1")
    circ2 = classical_to_quantum(vector2, "UGate_Angle_Embedding_2")
    circ = QuantumCircuit(2, 2)
    circ.append(circ1, [0])
    circ.append(circ2, [1])
    circ.cx(0, 1)
    circ.h(0)
    circ.measure(0, 0)
    circ.measure(1, 1)

    #print("exit bell test") # debug

    return circ


#Using Aer simulator

def circuit_simulator(circ, nshots):
    # Simulates a given Quantum Circuit
    
    # INPUTS:
    # circ: QuantumCircuit --- Quantum circuitry 
    
    # OUTPUTS:
    # counts: dict --- Result of Simulation
    
    #print("enter circuit simulator") # debug

    simulator = Aer.get_backend('aer_simulator')
    #Other options: 'aer_simulator_statevector', 'aer_simulator_density_matrix', 'aer_simulator_stabilizer', 'aer_simulator_matrix_product_state', 'aer_simulator_extended_stabilizer', 'aer_simulator_unitary', 'aer_simulator_superop', 'qasm_simulator'

    ### CAN ALSO SIMULATE WITH NOISE ###
    # https://qiskit.org/documentation/tutorials/simulators/2_device_noise_simulation.html, https://qiskit.org/documentation/apidoc/aer_noise.html 
    # from qiskit.providers.fake_provider import FakeVigo
    # device_backend = FakeVigo()
    # simulator = AerSimulator.from_backend(device_backend)


    qc_compiled = transpile(circ, simulator)

    #running and getting counts
    job_sim = simulator.run(qc_compiled, shots=nshots)
    result_sim = job_sim.result()
    counts = result_sim.get_counts()
    
    
    #print("exit circuit simulator with P(11) =" + str((counts.get('11',0))/nshots)) # debug
    
    #Returning the number of times that 11 occurs in simulation - proportional to probability of getting 11 - which is directly proportional to 3D euclidean distance  
    return counts.get('11',0)


def stereo_proj(data, r):

    # Transforms a given 2d data to its stereographic projection on sphere of radius r 

    # INPUTS:
    # data: np.array --- input 2d data
    # r: Desired radius of stereographic projection

    # OUTPUTS:
    # tx_init_pt: np.array --- 3d cartesian coordinates of sterographically projected data

    
    #print("enter stereographic projection") # debug


    #Transforming dataset to ISP
    d_x = (r*r)*2*data.real/(r*r + np.absolute(data)*np.absolute(data))
    d_y = (r*r)*2*data.imag/(r*r + np.absolute(data)*np.absolute(data))
    d_z = r*(-r*r + np.absolute(data)*np.absolute(data))/(r*r + np.absolute(data)*np.absolute(data))

    #Formatting data points
    tx_init_pt = np.column_stack((d_x, d_y, d_z))

    
    #print("exit stereographic projection") # debug


    return tx_init_pt


def spherical_cartesian_to_polar(data_in):

    
    #print("enter cartesian to polar") # debug


    
    theta = np.arctan2( np.sqrt(data_in[:,0]**2 + data_in[:,1]**2) , data_in[:,2] )
    phi = np.arctan2( data_in[:,1], data_in[:,0] )

    #Formatting angular data
    angular_data = np.row_stack((theta,phi))
    #angular_data = [theta,phi]
    #print(np.shape(angular_data))

    
    #print("exit cartesian to polar") # debug


    return angular_data

def spherical_cartesian_to_polar_single_element(data_in):
    
    theta = np.arctan2( np.sqrt(data_in[0]**2 + data_in[1]**2) , data_in[2] )
    phi = np.arctan2( data_in[1], data_in[0] )

    #Formatting angular data
    #angular_data = np.column_stack((theta,phi))
    angular_data = [theta,phi]
    #print(np.shape(angular_data))
    return angular_data



def qk_mean(k, data, fixed_centroids, maxNumIter, r, shots):
    # Quantum K-mean Algorithm

    # INPUTS:
    # k: int --- k of k-mean. Decides the initial number of centroids.
    # data: np.array ---
    # fixed_centroids --- this array sets initial given points as centroids 
    # maxNumIter: int --- Maximum number of iteration

    # OUTPUTS:
    # dataClusters: np.array --- Output Clusters of the algorithm

    
    # print("entering quantum k means") # debug -alonso



    # Clusters Before is used for understanding the changes between iterations
    dataClusters = np.zeros(len(data))
    clustersBefore = np.zeros(len(data))

    # For a point in centroids [x1,x2,x3] x1 and x2 are coordinates. x3 is the Centroid number ranging from 1 to k
    k, whocares = fixed_centroids.shape

        
    # Stereographic Embedded Centroids
    transformedCentroids = np.zeros([k,4])
    transformedCentroids[:,3] = np.arange(1, k+1)


    # Embedding data and initial centroids
    transformedData = stereo_proj(data, r)
    #print(np.shape(transformedData))
    #print(len(transformedData))
    transformedCentroids[:, [0, 1, 2]] = stereo_proj( fixed_centroids, r )
    #print(np.shape(transformedCentroids))
    


    didClustersChange = False
    numIter = 0

    
    


    while not didClustersChange and numIter < maxNumIter:

        
        # print("iteration of quantum k mean = " + str(numIter) + "\n now entering cluster assignment") # debug -alonso


        np.copyto(clustersBefore, dataClusters)

       
        # Assigning data points to different clusters
        dataClusters = cluster_assignment(      transformedData,        transformedCentroids,        dataClusters,        shots)

        # print("cluster assigned, iteration no " + str(numIter) ) # debug -alonso

        if np.array_equal(dataClusters, clustersBefore, equal_nan=False):
            didClustersChange = True

        # Deleting empty cluster centroids
        delList = []
        for i in np.arange(len(transformedCentroids)):
            if np.sum([dataClusters == transformedCentroids[i, 3]]) == 0:
                delList.append(i)
        centroids = np.delete(transformedCentroids, delList, 0)
        transformedCentroids = np.delete(transformedCentroids, delList, 0)

        # Finding new centroids from data by taking the mean of points in the clusters.
        for i in np.arange(len(transformedCentroids)):
            transformedCentroids[i,[0,1,2]] = data[dataClusters == transformedCentroids[i,3]].mean(axis = 0)                    
        

        numIter = numIter + 1       


    return dataClusters, centroids, numIter



def cluster_assignment(data, centroids, dataClusters, shots):
    # Assings clusters to each point in data
    
    # INPUTS:
    # data: np.array --- 
    # centroids: np.array(k,3) --- 
    # dataClusters: np.array(k,1) --- 
    # shots: int --- 
    # r: float --- 
    
    # OUTPUTS:
    # dataClusters: np.array --- Output Clusters after iteration
    # print("entered cluster assignment") #debug

    for point in range(0,len(data)): 
        
        #print("cluster assignment point no" + str(point))
        # calculating polar and azimuthal angles of data point and all centroids for stereograpic embedding
        ### To note: Conjugate of stereographic projection of data point is fed to calculate correct inner product ###  
        data_angles = spherical_cartesian_to_polar_single_element(data[point,:])
        tx_centroid_angles = spherical_cartesian_to_polar( centroids[:, [0, 1, 2]] )

        #print("cluster assignment stuck @ 1") # debug
                        
        #  Creating circuitry for every centroid.
        circs = [   [   bell_test( data_angles  ,   tx_centroid_angles[: , centroid] )   ]     for centroid in range(len(centroids))    ]

        #print("cluster assignment stuck @ 2") # debug

        # """
        # if point == 0 and numIter == 0:
        #     plt.figure(parallelCircs[0].decompose().draw(output='mpl', style={'backgroundcolor': '#EEEEEE'}))
        #     plt.show()
        # """
        
        results = [ circuit_simulator( circ , shots ) for circ in circs ] 

        #print("cluster assignment stuck @ 3") # debug
        
        # Finding the Clusters from the data
        
        dataClusters[point] = centroids[ np.argmin(results) , 3 ]

            
    return dataClusters



def bit_to_decimal(data):
    #takes an array of bits and maps it to an array of decimals

    data_aux = []
    for x in data:
        aux = 0
        for k in range(len(x)):
            aux= aux + x[k]*2**((len(x)-1)-k)
    
        data_aux.append(aux)
    return data_aux
