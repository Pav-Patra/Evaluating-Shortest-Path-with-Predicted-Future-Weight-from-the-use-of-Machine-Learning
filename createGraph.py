import Incremental_Shortest_Path as ISP     # my own defined Graph object for the ISP algorithm
from SP_NN_model import SP_NN as SPN        # my defined shortest path neural network model
import matplotlib.pyplot as plt
import keyboard, time
#import graphEmbedding as GE        # defined class that returns embedding of a graph
import random

"""
This file is created by Pav Patra


createGraph.py demonstrates how the Incremental Shortest Path algorithm alongside its Shortest 
Path Nerual Network used to return the shortest path between any 2 nodes for dynamic, undirected, 
weighted graphs in real time.
"""


# Create a graph 
g = ISP.Graph()

# setup for the trained Shortet Path model  
model = SPN()


print("As this algorithm executes continuous dynamic changes to the undirected, weighted graph, press [SPACE] to pause execution to fetch a shortest path distance")
rank = int(input("Enter the total number of nodes for a new graph:"))




g.generateGraph(rank)

print(g.graph)

g.drawGraph()    


g.incrementalShortestPath()
print(f"Number of nodes = {len(g.addedNodes)}")

model.declareFirstDataset(g)

plt.ion()

plt.show()

src = 0
dest = 4

print(f"\nShortest Path from node {src} to {dest} = {g.findShortestPath(src, src, dest)}")


print(f"Total number of nodes, starting from 0, is: {max(g.graph)}")

# an input can act as a block to prevent the previous plot from closing
fromNode= int(input("Enter the src node number you would like to use to identify a shortest path:"))
toNode = int(input("Enter the dst node number you would like to use to identify a shortest path:"))
print(f"\nShortest Path from node {fromNode} to {toNode} = {g.findShortestPath(fromNode, fromNode, toNode)}")


# CODE THAT RANDOMLY CHANGES EDGES ON THE GRAPH EVERY 5 SECONDS FOR A SPECIFIED NUMBER OF TIMES

respone = 1

totalSP = 0
totalWrong = 0


# This block of code continuously updates a random edge weight on the graph every 3 seconds until an interrupt occurs (holding space)
# after this interrupt, the user can view the graph and select a shortest path between 2 nodes
# the user then has the option to continue with graph execution or to stop
while respone == 1:

    while not keyboard.is_pressed('space'):   # if the shortest path is desired to be displayed by the user, the user must hold down 'space'
                                              # this is the case due to time.sleep() where the system cannot read any key press during this time

        if(random.randint(1,3) == 2):    # 1/3  probability of the graph randomly changing on each 1 second iteration
            g.changeGraph()
        
        

        time.sleep(1)

    plt.clf()
    plt.cla()
    g.drawGraph()

    plt.show()

    # there is no way of determining whether the graph requires updating with the ISP algorithm or not

    # instead of executing this directly, utilise SPNM
    # SPNM is a model which takes in graph g, a src node and a destination node
    # initialise this model and use it
    # if the return value is non-zero, the graph needs updating
    
    
    fromNode= int(input("Enter the src node number you would like to use to identify a shortest path:"))
    toNode = int(input("Enter the dst node number you would like to use to identify a shortest path:"))

    totalSP += 1

    # run the Shortest Path NN model to establish edge weight change between src and dst nodes
    change = model.run(g, src=fromNode, dst=toNode)


    print(f"Calculated Change value: {change}")

    if(change >= 1e-6):
            print("Execute ISP Update")
            g.incrementalShortestPath()
            model.declareFirstDataset(g)

    originalLength = g.findShortestPath(fromNode, fromNode, toNode)

    print(f"\nShortest Path from node {fromNode} to {toNode} = {originalLength}")



    g.incrementalShortestPath()

    testLength  = g.findShortestPath(fromNode, fromNode, toNode)


    if (testLength != originalLength) and (change <= 1e-6):
         totalWrong += 1
         print("FIX ERROR")
         model.declareFirstDataset(g)
    
    print("As this algorithm executes continuous dynamic changes to the undirected, weighted graph, press [SPACE] to pause execution to fetch a shortest path distance")
    
    respone = int(input("Would you like graph execution to continue? 1:YES 2:NO"))

# recompute the shortest path n-ary trees for each node after the graph change
# hence, for each edge change in the grapha, dijkstra is computed M times (where M is the number of nodes in the graph)



fig = plt.gcf()
fig.clf()
g.drawGraph()
fig.show()


respone = int(input("Would you like graph execution to continue? 1:YES 2:NO"))


totalRight = totalSP - totalWrong
print(f"Accuracy of model is: {(totalRight/totalSP)*100}%")
