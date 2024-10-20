import Incremental_Shortest_Path as ISP     # my own defined Graph object for the ISP algorithm
from SP_NN_model import SP_NN as SPN        # my defined shortest path neural network model
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import bellman_ford, dijkstra   # benchmarkin well-known shortest path algorithms
from time import perf_counter
import random
import os



"""
This file is created by Pav Patra

test_graph.py is the testing suite for this project

Testing is maintained for both Incremental_Shortest_Path and SP_NN_Model Python files

This file also contains the benchmark testing results of the developed ISP algorithm against Dijkstra's 
and the Bellman-Ford algorithm

To run this file, simply execute: python test_graph.py
"""


# test file to carry out a series of unit tests to test the Graph class functions in Incremental_Shortest_Path.py 

def test_test():
    assert(5+6) == 11


test_g = ISP.Graph()
test_m = SPN() 
# general test case for the Increental Shortest Path Algorithm 
def test_ISPalgo():
    

    test_g.generateGraph(100)

    test_g.drawGraph()

    test_g.incrementalShortestPath()

    src = 2
    dst = 4

    nxShortestPath = nx.shortest_path_length(test_g.draw_graph, source=src, target=dst, weight='weight') 

    print(f"\nShortest Path from node {src} to {dst} with my algo = {test_g.findShortestPath(src, src, dst)}")
    # test_g.findShortestPath(src, src, dst) is a simple tree search, hence it can be executed multiple times with a low run time

    print(f"Test result is = {nxShortestPath}")

    assert(test_g.findShortestPath(src, src, dst)) == nxShortestPath



# test case the minimum and maximium node in the ISP algorithm
def min_max_case():
    test_g.drawGraph()
    test_g.incrementalShortestPath()

    src = 0
    dst = max(test_g.graph)

    nxShortestPath = nx.shortest_path_length(test_g.draw_graph, source=src, target=dst, weight='weight') 

    print(f"\nShortest Path from node {src} to {dst} with my algo = {test_g.findShortestPath(src, src, dst)}")

    print(f"Test result is = {nxShortestPath}")

    assert(test_g.findShortestPath(src, src, dst)) == nxShortestPath



# original issue raised IndexErrors when attempting to get the weight of the shortest path from the maximum node to the minimum node
def max_min_case():
    test_g.drawGraph()
    test_g.incrementalShortestPath()

    src = max(test_g.graph)
    dst = 0

    nxShortestPath = nx.shortest_path_length(test_g.draw_graph, source=src, target=dst, weight='weight') 

    print(f"\nShortest Path from node {src} to {dst} with my algo = {test_g.findShortestPath(src, src, dst)}")

    print(f"Test result is = {nxShortestPath}")

    assert(test_g.findShortestPath(src, src, dst)) == nxShortestPath





def test_Change():
    
    print("")
    print("Change")
    test_g.changeGraph()

    test_g.drawGraph()

    test_g.incrementalShortestPath()

    src = 2
    dst = 4

    nxShortestPath = nx.shortest_path_length(test_g.draw_graph, source=src, target=dst, weight='weight') 

    print(f"\nShortest Path from node {src} to {dst} with my algo = {test_g.findShortestPath(src, src, dst)}")
    # test_g.findShortestPath(src, src, dst) is a simple tree search, hence it can be executed multiple times with a low run time

    print(f"Test result is = {nxShortestPath}")

    assert(test_g.findShortestPath(src, src, dst)) == nxShortestPath


# add new unit test that tests the accuracy of the SP NN on a live dynamic graph of rank=n nodes
def test_pathAccuracy_model():
    # declare model
    print(test_g.graph)

    test_m.declareFirstDataset(test_g)

    # take test_g.drawGraph as base graph
    # get random src and dst nodes
    src = 2
    dst = 35

    originalDistance = test_g.findShortestPath(src,src,dst)

    # execute a series of graph changes in an attempt to alter the distance between src and dst
    for _ in range(50):
        test_g.changeGraph()
    
    change = test_m.run(test_g, src=src, dst=dst)

    if(change >= 1e-6):
            print("Execute ISP Update")
            test_g.incrementalShortestPath()
            test_g.drawGraph()                  # update nx graph
            test_m.declareFirstDataset(test_g)

    actualDistance = nx.shortest_path_length(test_g.draw_graph, source=src, target=dst, weight='weight')

    print(f"OriginalDistance: {originalDistance}")
    print(f"actualDistance: {actualDistance}")
    print(f"ISP answer: {test_g.findShortestPath(src, src, dst)}")

    assert(test_g.findShortestPath(src, src, dst)) == actualDistance



def test_runAccuracy_model():
    # test to see whether each run is necessary
    # in other words, has the model accurately predicted to run the ISP algo due to a path weight change
    # count these instances
    test_m.declareFirstDataset(test_g)

    totalRuns = 0
    correctRun = 0

    countISPRun = 0
    countNoISPRun = 0

    src = 4
    dst = 17

    errorString = "start model error displayu\n"

    for _ in range(30):
        ispRun = False
        totalRuns += 1
        originalDistance = test_g.findShortestPath(src, src, dst)

        print(f"Original weighted distance of shortest path between {src} and {dst} is: {originalDistance}")

        for _ in range(50):
            test_g.changeGraph()

        change = test_m.run(test_g, src=src, dst=dst)

        if(change >= 1e-6):
            print("Execute ISP Update")
            ispRun = True
            countISPRun += 1
            test_g.incrementalShortestPath()
            test_g.drawGraph()                  # update nx graph
            test_m.declareFirstDataset(test_g)
        else:
            countNoISPRun += 1

        newDistance = test_g.findShortestPath(src,src,dst)
        print(f"New weighted distance of shortest path between {src} and {dst} is: {newDistance}")

        actualDistance = nx.shortest_path_length(test_g.draw_graph, source=src, target=dst, weight='weight')

        # correct run has been performed only if the isp algorithm has been executed and the new distance is different from the old one
        # correct run also has been performed if isp alogrithm has not been run and the original distance is equal to the new distance
        if actualDistance == newDistance and originalDistance != newDistance and ispRun is True:
            correctRun += 1
        elif actualDistance == newDistance and originalDistance == newDistance and ispRun is False:
            correctRun += 1
        else:
            errorString += "INCORRECT  RUN\n"
            errorString += "ispRun is: " + str(ispRun) + "\n"
            errorString += "originalDistance is " + str(originalDistance) + " and newDistance is: " + str(newDistance) + "\n"
            
    accuracy = correctRun / totalRuns

    print(f"Number of runs with ISP: {countISPRun}")
    print(f"Number of runs without ISP: {countNoISPRun}")
    print(f"Number of correct ISP/noISP runs: {correctRun}")
    print(f"Accuracy of running ISP algo when needed: {accuracy}")
    print("")
    print(errorString)

    # at least 70% of runs must be correct in running the ISP alogorithm or not
    assert accuracy >= 0.7



# create new set of unit tests which compraes the results and run time of retuening the qeight of the shortest path using this
# ISP algorithm with other algorithms such as A* and Bellman-Ford


# first test run comparison is an instance of BellmanFord
def bellmanTest():
    src = 12
    dst = 31
    test_m.declareFirstDataset(test_g)

    rank = len(test_g.addedNodes)

    ISPTimes = np.empty([10], dtype=float)

    BFTimes = np.empty([10], dtype=float)

    testResultsWrite = "ISP vs BF Test Times:\nISP Times:"

    # run 10 random path search instances for both algorithms and time them using timeit
    # before each path search, 50 random edges must be changed

    # 10 ISP algo runs

    for i in range(10):
        start = perf_counter()

        for _ in range(50):
            test_g.changeGraph()

        src = random.randint(0, rank-1)
        dst = random.randint(0, rank-1)

        change = test_m.run(test_g, src=src, dst=dst)

        if(change >= 1e-6):
            print("Execute ISP Update")
            test_g.incrementalShortestPath()
            test_m.declareFirstDataset(test_g)

        distance = test_g.findShortestPath(src,src,dst)

        print(f"Distance between {src} and {dst} with ISP is: {distance}")

        end = perf_counter()

        ISPTimes[i] = end - start

  

    # 10 BF algo runs
        
    for i in range(10):
        start = perf_counter()

        for _ in range(50):
            test_g.changeGraph()

            # convert test_g into a rank x rank matrix
            bfMatrix = np.zeros((rank, rank))

            for u in range(rank):
                for v, weight in test_g.graph[u]:
                    bfMatrix[u][v] = weight

            calcBFMatrix = bellman_ford(csgraph=csr_matrix(bfMatrix), directed=False, return_predecessors=False)

        src = random.randint(0, rank-1)
        dst = random.randint(0, rank-1)

        # convert test_g into a rank x rank matrix
        bfMatrix = np.zeros((rank, rank))

        for u in range(rank):
            for v, weight in test_g.graph[u]:
                bfMatrix[u][v] = weight

        calcBFMatrix = bellman_ford(csgraph=csr_matrix(bfMatrix), directed=False, return_predecessors=False)

        print(f"Distance between {src} and {distance} using BF is: {calcBFMatrix[src][dst]}")

        end = perf_counter()
        BFTimes[i] = end - start

    print("ISP Times:")
    print(ISPTimes)
    print("")

    print("BF Times:")
    print(BFTimes)
    print("")

    for time in ISPTimes:
        testResultsWrite = f"{testResultsWrite}\n{time}"
    testResultsWrite = f"{testResultsWrite}\n\nBF Times:"
    for time in BFTimes:
        testResultsWrite = f"{testResultsWrite}\n{time}"

    # save results to a text file called "results.txt" in the cwd
    current_dir = os.getcwd()

    # define the file path
    file_path = os.path.join(current_dir, "results.txt")

    # open new file in write mode
    with open(file_path, "w") as file:
        file.write(testResultsWrite)
    file.close()


    
    # PLOT RESULTS
    plt.figure(figsize=(10,5))

    # width of bar
    width = 0.3

    # number of data points
    n = 10
    # posiotion of bars
    ind = np.arange(n)

    plt.bar(ind, ISPTimes, width, label='ISP Time')
    plt.bar(ind + width, BFTimes, width, label='BF Time')

    plt.xlabel('Iteration', fontsize=15, fontweight='bold')
    plt.ylabel('Elapsed Execution Time (s)', fontsize=15, fontweight='bold')
    plt.title('Comparison of ISP vs Bellman-Ford Runtimes')

    plt.ylim(0, 2)

    plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'), fontsize=15, fontweight='bold')

    plt.legend(loc='best', fontsize='large', title_fontsize='15', prop={'weight':'bold'})
    
               
    

    assert 1 == 1



# second test run comparison is an instance of Dijkstra's
def dijkstraTest():
    src = 12
    dst = 31
    test_m.declareFirstDataset(test_g)

    rank = len(test_g.addedNodes)

    ISPTimes = np.empty([10], dtype=float)

    dijkTimes = np.empty([10], dtype=float)

    testResultsWrite = "\n\nISP vs Dijkstra Test Times:\nISP Times:"

    # run 10 random path search instances for both algorithms and time them using timeit
    # before each path search, 50 random edges must be changed

    # 10 ISP algo runs

    for i in range(10):
        start = perf_counter()

        for _ in range(50):
            test_g.changeGraph()

        src = random.randint(0, rank-1)
        dst = random.randint(0, rank-1)

        change = test_m.run(test_g, src=src, dst=dst)

        if(change >= 1e-6):
            print("Execute ISP Update")
            test_g.incrementalShortestPath()
            test_m.declareFirstDataset(test_g)

        distance = test_g.findShortestPath(src,src,dst)

        print(f"Distance between {src} and {dst} with ISP is: {distance}")

        end = perf_counter()

        ISPTimes[i] = end - start
        
        


    # 10 BF algo runs
        
    for i in range(10):
        start = perf_counter()

        for _ in range(50):
            test_g.changeGraph()

             # convert test_g into a rank x rank matrix
            dijMatrix = np.zeros((rank, rank))

            for u in range(rank):
                for v, weight in test_g.graph[u]:
                    dijMatrix[u][v] = weight

            calcDijMatrix = dijkstra(csgraph=csr_matrix(dijMatrix), directed=False, return_predecessors=False)

        src = random.randint(0, rank-1)
        dst = random.randint(0, rank-1)

        # convert test_g into a rank x rank matrix
        dijMatrix = np.zeros((rank, rank))

        for u in range(rank):
            for v, weight in test_g.graph[u]:
                dijMatrix[u][v] = weight

        calcDijMatrix = dijkstra(csgraph=csr_matrix(dijMatrix), directed=False, return_predecessors=False)

        print(f"Distance between {src} and {distance} using BF is: {calcDijMatrix[src][dst]}")

        end = perf_counter()
        dijkTimes[i] = end - start


    print("ISP Times:")
    print(ISPTimes)
    print("")

    print("Dijkstra Times:")
    print(dijkTimes)
    print("")

    for time in ISPTimes:
        testResultsWrite = f"{testResultsWrite}\n{time}"
    testResultsWrite = f"{testResultsWrite}\n\nDijkstra Times:"
    for time in dijkTimes:
        testResultsWrite = f"{testResultsWrite}\n{time}"


    # save results to a text file called "results.txt" in the cwd
    current_dir = os.getcwd()

    # define the file path
    file_path = os.path.join(current_dir, "results.txt")

    # open new file in write mode
    with open(file_path, "a") as file:
        file.write(testResultsWrite)
    file.close()

    # PLOT RESULTS
    plt.figure(figsize=(10,5))

    # width of bar
    width = 0.3

    # number of data points
    n = 10
    # posiotion of bars
    ind = np.arange(n)

    plt.bar(ind, ISPTimes, width, label='ISP Time')
    plt.bar(ind + width, dijkTimes, width, label='Dijkstra Time')

    plt.xlabel('Iteration', fontsize=15, fontweight='bold')
    plt.ylabel('Elapsed Execution Time (s)', fontsize=15, fontweight='bold')
    plt.title('Comparison of ISP vs Dijkstra Runtimes')

    plt.ylim(0, 2)

    plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'), fontsize=15, fontweight='bold')

    plt.legend(loc='best', fontsize='large', title_fontsize='15', prop={'weight':'bold'})    
    

    assert 1 == 1


"""
    modelRunTimeTest()
    Method to test the run time difference between running the Incremental Shortest Path algorithm with the 
    ShortestPath NN and without this model.
    Results plotted onto a graph.
"""
def modelRunTimeTest():
    test_g = ISP.Graph()
    test_m = SPN() 
    test_g.generateGraph(100)
    test_g.incrementalShortestPath()
    test_m.declareFirstDataset(test_g)

    src = 12
    dst = 31

    rank = len(test_g.addedNodes)

    ISPTimes = np.empty([10], dtype=float)

    ISPModelTimes = np.empty([10], dtype=float)

    testResultsWrite = "\n\nISP with model vs without model Times:\nwith model Times:"

    # run 10 random path search instances for both algorithms and time them using timeit
    # before each path search, 50 random edges must be changed

    # 10 ISP algo runs with the model

    for i in range(10):
        start = perf_counter()

        for _ in range(50):
            test_g.changeGraph()

        src = random.randint(0, rank-1)
        dst = random.randint(0, rank-1)

        change = test_m.run(test_g, src=src, dst=dst)

        if(change >= 1e-6):
            print("Execute ISP Update")
            test_g.incrementalShortestPath()
            test_m.declareFirstDataset(test_g)

        distance = test_g.findShortestPath(src,src,dst)

        print(f"Distance between {src} and {dst} with ISP is: {distance}")

        end = perf_counter()

        ISPModelTimes[i] = end - start


    # 10 ISP algo runs without the model 
    for i in range(10):
        start = perf_counter()

        for _ in range(50):
            test_g.changeGraph()

        src = random.randint(0, rank-1)
        dst = random.randint(0, rank-1)

        test_g.incrementalShortestPath()

        distance = test_g.findShortestPath(src,src,dst)

        print(f"Distance between {src} and {dst} with ISP is: {distance}")

        end = perf_counter()

        ISPTimes[i] = end - start    

    print("ISP Times with Model:")
    print(ISPModelTimes)
    print("")

    print("ISP Times without Model:")
    print(ISPTimes)
    print("")

    for time in ISPModelTimes:
        testResultsWrite = f"{testResultsWrite}\n{time}"
    testResultsWrite = f"{testResultsWrite}\n\nwithout model Times:"
    for time in ISPTimes:
        testResultsWrite = f"{testResultsWrite}\n{time}"

    # save results to a text file called "results.txt" in the cwd
    current_dir = os.getcwd()

    # define the file path
    file_path = os.path.join(current_dir, "results.txt")

    # open new file in write mode
    with open(file_path, "a") as file:
        file.write(testResultsWrite)
    file.close()


    # PLOT RESULTS
    plt.figure(figsize=(10,5))

    # width of bar
    width = 0.3

    # number of data points
    n = 10
    # posiotion of bars
    ind = np.arange(n)

    plt.bar(ind, ISPModelTimes, width, label='ISP Model Time')
    plt.bar(ind + width, ISPTimes, width, label='ISP non-Model Time')

    plt.xlabel('Iteration', fontsize=15, fontweight='bold')
    plt.ylabel('Elapsed Execution Time (s)', fontsize=15, fontweight='bold')
    plt.title('Comparison of ISP with Model vs ISP without Model Runtimes')

    plt.ylim(0, 2)

    plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'), fontsize=15, fontweight='bold')

    plt.legend(loc='best', fontsize='large', title_fontsize='15', prop={'weight':'bold'})    
    

    assert 1 == 1




if __name__ == "__main__":
    test_test()
    try:
        test_ISPalgo()
        
    
    except AssertionError:
        # discovered error from .generateGraph() where additional edge weights for assigned edges are being added, duplicating edges with different weights as a result 
        test_g.printTree()
        print("")
        print(test_g.graph)
        plt.ion()
        plt.show()
        end = int(input("Assertion Failure"))
    test_Change()
    min_max_case()
    max_min_case()

    test_pathAccuracy_model()

    # test_runAccuracy_model()

    bellmanTest()
    dijkstraTest()
    modelRunTimeTest()

    print("Everything Passed")

    plt.show()



    

