import Incremental_Shortest_Path as ISP     # my own defined Graph object for the ISP algorithm
import matplotlib.pyplot as plt
import keyboard, time
#import graphEmbedding as GE        # defined class that returns embedding of a graph
import random
import os


# Create a graph 

g = ISP.Graph()



g.generateGraph(10)

print(g.graph)

#print(g.addedNodes)
#print(g.addedEdges)


#g.printTree()

g.drawGraph()    


g.incrementalShortestPath()

plt.ion()

plt.show()

src = 0
dest = 4

print(f"\nShortest Path from node {src} to {dest} = {g.findShortestPath(src, src, dest)}")

#print(f"Current latest node is {max(g.graph)}")

print(f"Total number of nodes, starting from 0, is: {max(g.graph)}")
# an input can act as a block to prevent the previous plot from closing
fromNode= int(input("Enter the src node number you would like to use to identify a shortest path:"))
toNode = int(input("Enter the dst node number you would like to use to identify a shortest path:"))
print(f"\nShortest Path from node {fromNode} to {toNode} = {g.findShortestPath(fromNode, fromNode, toNode)}")



#####################################################################################################################################
# CODE THAT RANDOMLY CHANGES EDGES ON THE GRAPH EVERY 5 SECONDS FOR A SPECIFIED NUMBER OF TIMES

respone = 1

# while True:
#     print("working")
#     if random.randint(0,30) == 15:
#         g.changeGraph()
#         g.incrementalShortestPath()
#     if keyboard.read_key() == 'space':
#         break

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
    g.incrementalShortestPath()
    fromNode= int(input("Enter the src node number you would like to use to identify a shortest path:"))
    toNode = int(input("Enter the dst node number you would like to use to identify a shortest path:"))

    print(f"\nShortest Path from node {fromNode} to {toNode} = {g.findShortestPath(fromNode, fromNode, toNode)}")

    respone = int(input("Would you like graph execution to continue? 1:YES 2:NO"))


#######################################################################################################################################



#g.changeGraph()

#g.addAdditionalNode(0)


# recompute the shortest path n-ary trees for each node after the graph change
# hence, for each edge change in the grapha, dijkstra is computed M times (where M is the number of nodes in the graph)

#g.incrementalShortestPath()


#inp= int(input("Would you like a change to the graph? 1:YES 2:NO"))

fig = plt.gcf()
fig.clf()
g.drawGraph()
fig.show()


# generate embedding for the graph

# embedding = GE.getEmbedding(g.draw_graph)   # pass in  the networkx representation of graph g to .getEmbedding()

# fig = plt.gcf()
# fig.clf()
# plt.scatter(embedding[:,0], embedding[:,1])
# fig.show()

respone = int(input("Would you like graph execution to continue? 1:YES 2:NO"))
