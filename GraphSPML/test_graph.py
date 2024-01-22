import Incremental_Shortest_Path as ISP     # my own defined Graph object for the ISP algorithm
import networkx as nx
import matplotlib.pyplot as plt


# test file to carry out a series of unit tests to test the Graph class functions in Incremental_Shortest_Path.py 

def test_test():
    assert(5+6) == 11


test_g = ISP.Graph()
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


if __name__ == "__main__":
    test_test()
    try:
        # either networkx is wrong or my algo is wrong
        test_ISPalgo()
        print("Everything Passed")
    
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

    

