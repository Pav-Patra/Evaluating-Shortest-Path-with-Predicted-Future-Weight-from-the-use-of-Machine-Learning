from collections import defaultdict    # used to create the adjacency matrix of node connections with fast look-up times
import networkx as nx
import random    # used for random number generation in graph plots
import heapq     # used to initialise a priority queue for Dijkstra's
from ctypes import Array



# define a tree class
class Node():

    def __init__(self, name, weight: int):
        self.children = []
        self.child = (name, weight)

    def addChild(self, name, weight):
        self.children.append((name, weight))

    def returnChildren(self):
        return self.children
    
    def getName(self):
        return self.child


# class represents the construction of a graph using an adjacency list
class Graph:

    #constructor
    def __init__(self):

        # default dictionary to store graph
        # must conver to an adjacency matrix that holds stores weight in each index
        # x's and y's of matrix represents node numbers
        self.graph = defaultdict(list)

        # list of rooted Nodes
        self.trees = [] 

        self.addedEdges = []
        self.addedNodes = [] 

        # used to hold recent edge weight changes to draw red
        self.recentChanges = []

        # declare the network that will be used to plot the graph onto a canvas
        self.draw_graph = nx.Graph()


    # Function to add an edge
    def addEdge(self, u: int, v: int, weight: int):
        self.graph[u].append((v, weight))
        if( (u, weight) not in self.graph[v] ):         # form the adjacency matrix by adding the weight for the indices 
            self.graph[v].append((u, weight))           # reversed as well (backwards directrion in an uindirected graph)
            

    # second parameter g must be of type Graph
    def generateGraph(self, num_nodes: int):
        # automatic way of generating the graph g with a large number of nodes and edges

        # this graph will ensure the graph contains double the number of edges than nodes

        totalNodes = num_nodes        # add each of these stated totalNodes to the graph, starting from zero
        for i in range(totalNodes):
            #print("")
            # random node values
            x = i
            y = random.randint(0,totalNodes-1)
            while x == y or (x, y) in self.addedEdges or (y, x) in self.addedEdges:     # don't add the same nodes connected to each other or edges already contained in the graph
                # recalculate the edge node commections
                x = random.randint(0,totalNodes)
                y = random.randint(0, totalNodes)

            # fix the case where the graph is unconnected. Graph must be connected
            
            #print(f"addedNodes list: {self.addedNodes}")
            #print(f"For x: {x not in self.addedNodes and self.addedNodes != []}")
            #print(f"Number of added nodes = {len(self.addedNodes)}")
        
            # if node x is not already discovered and the list of visited nodes is not empty, ensure node x is also connected to a random node in the already drawn graph
            if x not in self.addedNodes and len(self.addedNodes) > 0:     # change len(self.addedNodes) > 0 to ensure all nodes are connected, however this significantly slows down performance. 
                #print("fix x")
                tempNode = random.choice(self.addedNodes)
                self.addEdge(tempNode, x, weight=random.randint(1, 10))
                self.addedNodes.append(x)
                self.addedEdges.append((tempNode, x))
            elif self.addedNodes == []:          # since the list of visisted nodes is empty, it is impossible to assign x to an existing node
                self.addedNodes.append(x)

            #print(f"For y: {y not in self.addedNodes and self.addedNodes != []}")
            # if node y is not already discovered and the list if visited nodes is not empty, ebsure node y is also connected to a random node in the already drawn graph
            if y not in self.addedNodes and len(self.addedNodes) > 0:      # change len(self.addedNodes) > 0 to ensure all nodes are connected, however this significantly slows down performance. 
                #print("fix y")
                tempNode = random.choice(self.addedNodes)
                self.addEdge(tempNode, y, weight=random.randint(1, 10))
                self.addedNodes.append(y)
                self.addedEdges.append((tempNode, y))
            elif self.addedNodes == []:          # since the list of visisted nodes is empty, it is impossible to assign x to an existing node
                self.addedNodes.append(y)

            # now add edge connection from new node to node y, regardless of whether it had already existed in the graph or not
            # final check to not duplicate edges with different weights
            if (x,y) not in self.addedEdges and (y,x) not in self.addedEdges:
                self.addEdge(x, y, weight=random.randint(1, 10))
                self.addedEdges.append((x,y))

      
    def drawGraph(self):
        
        self.draw_graph = nx.Graph()

        #print(self.graph)

        x = 0
        y = 0

        # Node number start from 0
        numNodes = max(self.graph) + 1

        #print(f"Nummber of nodes is {numNodes} in DG")


        for i in range(numNodes):
            self.draw_graph.add_node(i, pos=(x,y))
            #print(f"Adding node {i}")
            # x = random.randint(-50,100)
            # y = random.randint(-50,100)

        # draw all edges
        #print(self.graph[0][0][0])
        for i in range(numNodes):
            for s in self.graph[i]:
                self.draw_graph.add_edge(i,s[0], weight=s[1])    # destination node storedin first index of tuple
                #print(f"Edge: ({i}, {s[0]})")

        #print(f"Nummber of nodes is {numNodes} in DG")

        pos = nx.spring_layout(self.draw_graph, seed=7)     # positions for all nodes - seed for reproducibility
        #pos = nx.get_node_attributes(self.draw_graph, 'pos')
        nx.draw(self.draw_graph, pos, with_labels=True)
        #nx.draw(self.draw_graph, pos, edgelist = [self.addedEdges[0]], edge_color="tab:red", with_labels=True)

        # mark changed edges in red
        nx.draw(self.draw_graph, pos, edgelist = self.recentChanges, edge_color="tab:red", with_labels=True)

        # clear recent changes
        self.recentChanges = []

        labels = nx.get_edge_attributes(self.draw_graph,'weight')
        nx.draw_networkx_edge_labels(self.draw_graph, pos, edge_labels=labels,)
        #print(f"Nummber of nodes is {numNodes} in DG")


    def changeGraph(self):
        randomEdge = random.choice(self.addedEdges)
        x = randomEdge[0]
        y = randomEdge[1]

        # from g.graph, get the index of tuple for the connected edge between x and y
        # ideally change this weight to the new weight


        # Get index of tuple in the dictionary of x coords to y coords in g.graph
        # This allows us to extract the old weight from g.graph and update it for the new weight
        tupleIndexX = [s[0] for s in self.graph[x]].index(y)
        tupleIndexY = [s[0] for s in self.graph[y]].index(x)
        # fetch old weight
        oldWeight = self.graph[x][tupleIndexX][1]
        print(f"{self.graph[x]}")

        print(f"Random edge: ({x}, {y}) with weight = {oldWeight}")

        # generate new weight
        newWeight = random.randint(0, 30)

        # assign new weight to old tuple (python does not support tuple item assignment so the tuple must be generated again)
        # update in the reverse direction as well since weight between two edges is equal for both indexes x and y in the graph adjacency matrix
        newTuplex = (y, newWeight)
        newTupley = (x, newWeight)
        self.graph[x][tupleIndexX] = newTuplex
        self.graph[y][tupleIndexY] = newTupley

        # record change
        self.recentChanges.append((x,y))


        print(f"Edited graph: {self.graph[x]}")


    # Function to add a new node to the already existing graph
    # the function calculates the next node number by taking the existing total number of nodes
    # the function requirest a 'connect' parameter which connects the new node to an existing node in the current graph
    def addAdditionalNode(self, connect: int):

        newNode = max(self.graph)+1
        self.addedNodes.append(newNode)
        self.addedEdges.append((connect, newNode))
        print(f"Added new node {newNode} connected to 0 with random wetght")
        self.addEdge(connect, newNode, random.randint(0, 30))


    
    # function to delete a specified edge 'edg' from the graph 'self.graph'
    # the purpose of this fucntion is to investigate how the algorithm can recompute
    # shortest paths after edge deletions
    def deleteEdge(self, node1: int, node2: int):
        # parameters are two nodes where there exists an edge connection between them

        print(self.addedNodes)

        print(node1 in self.addedNodes)
        print(node2 in self.addedNodes)
        print((node1, node2) in self.addedEdges)
        print((node2, node1) in self.addedEdges)
        
        # firstly check if edge exists
        if node1 in self.addedNodes and node2 in self.addedNodes and ((node1, node2) in self.addedEdges or (node2, node1) in self.addedEdges):
            print("Edge exists")
        else:
            print("Edge does not exist")
 

    # Function to print a BFS of graph
    def BFS(self, s: int):

        # Mark all vertices as not visited
        visited = [False] * (max(self.graph) + 1)

        

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        print(f"Breadth First Search on the graph from node {s}:")

        while queue:

            # Dequeue a vertex from queue and print it
            s = queue.pop(0)
            print(s, end=" ")

            # Get all adjacent vertices of the dequeued vertex s. If an adjacent has not been visited, then mark it visited and enqueue it
            for i in [x[0] for x in self.graph[s]]:       # Since self.graph stores a list of tuples, get the first element of each selected tuple this way (dest node)
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        print("")     # newline


    def printTree(self):
        # construct shortest path tree for src node
        count = 0

        for i in self.trees:
            print("\n\n")
            print(f"Tree Rooted at {count}")
            count+=1
            for s in i:
                if s is not None:
                    print("")
                    print(f"src name: {s.child[0]} weight: {s.child[1]}")
                    print("Children: ")
                    print(s.children)
            # for x in i.children:
            #     print(f"(src: {x[0]}, dist: {x[1]})")



    # a priority queue (initialised with the heap library) is effective here since a pq can store the minimum distance adjacent node at the front of the queue, which is the requirement of Dijkstra's algorithm
    # however, a downside to the pq is that it doesn't support the decreasing key operation. A solutiojn to this is to insert more than one copy of a key each time it is updated since we only consider the minimum instance 
    # and ignore all others.
    # the purpose of using a heap/priority queue is to achieve a time complexity of O(E*logV) as there will be at most O(E) vertices in the priority queue and O(logE) is the same as O(logV)
    # the time complexity of an adjacency matrix implementation of Dijkstra's algorithm is O(V^2)
    def dijkstraSP(self, src: int) -> Array:       # this function prints thje shortest path from a given src to all other nodes
        # utilise a heap which uses a priority queue to store nodes that are being preprocessed
        #  
        pq = []
        heapq.heappush(pq, (0, src))     # the distance at the src is always 0



        # create vector that initiialises all distances as infinity
        dist = [float('inf')] * (max(self.graph) + 1) 
        dist[src] = 0

        #print(dist)

        #store visited paths
        # helps to purune previously discovered paths which aren't the shortest
        visited = [False] * (max(self.graph) + 1)

        #print(visited)
 
        # attempt to generate shortest path tree here
        tree_nodes = [None] * (max(self.graph) + 1)         # each node in the tree stores a node name from the graph and its sums of weight from src to selected node
        tree_nodes[src] = Node(src, dist[src])        # initiallise the root node and store in its index in the array

        visited[src] = True
        
        while pq:
            # First node in pair is the minimum distance node. Extract from pq
            # node label is stored in second index of tuple
            # print(pq)
            distN, u = heapq.heappop(pq)

            for v, weight in self.graph[u]:   # selects all nodes connected to u
                #print(f"{u} to {v} with weight {weight}")
                # if there is a shorter path to v through u
                if dist[v] > dist[u] + weight:
                    newWeight = dist[u] + weight
                    # update distance of v
                    dist[v] = newWeight

                    # prune previously calculated paths to v if any exist
                    if visited[v]:    # only prune if selected node has previously been visited
                        for x in range(max(self.graph)):     
                            if(visited[x] and v != x):              # only check node if it has been visited and not equal to the selected node
                                # filter out all occurences of node v in each of the visited nodes since a shorter path to v has been established 
                                tree_nodes[x].children = list(filter(lambda s : s[0] != v, tree_nodes[x].returnChildren()))
                    else:
                        visited[v] = True

                    heapq.heappush(pq, (dist[v], v))
                    tree_nodes[v] = Node(v, dist[v])
                    tree_nodes[u].addChild(v, dist[v])

        # add newly constructed tree to the list of rooted shortest paths
        self.trees[src] = tree_nodes      


        # print shortest distances from stated src node
        # print(f"Shortest path distances from src {src} to stated nodes:")    STDOUT 
        return dist


    # function to identify the shortest path in a dybnamic graph using the incremental algorithm 
    def incrementalShortestPath(self):
        # the initial basis of this algorithm will be to run dijkstraSP for each node in the graph to construct the shortest path trees for each node
        # the shortest path tree for each node will be constructed
        # an update algorithm will be implemented to alter these trees in case of: edge insertions, edge deletions, change in edge weights

        dists = []          # list of distance lists for each node
        # ensure tree nodes account for final maximum node in the graph
        self.trees = [None] * (max(self.graph) + 1) 
 
        for i in range(max(self.graph) + 1):
            temp = self.dijkstraSP(i)
            dists.append(temp)

        # potentially construct the shortest path tree in dijksraSP



    # function to print the shortest path between src to dest
    # for a stated src node, fecth the rooted shortest path tree from 'trees' and calculate the shortest path sum 
    # from src to dest
    def findShortestPath(self, root: int, src: int, dest: int):
        # root - ensures that the correct rooted tree is selected on each recursive call of the function
        # src - the currently selected node on the path between root to dest
        # dest - the final node which is being searched for

        
        selectTree = self.trees[root]    # select the rooted shortest path tree
                                         # issue here when selecting the last node (greates nuimbered node) as the src node (Index Error: list index out of range)

        # if the destination node has been selected after a recursive call, return the selected node's stored summed weight
        # this represents the shortest path between root to dest

        #print(src, end=" ")
        if(src == dest):
            return selectTree[src].child[1]
        
        #print(f"Test: {selectTree[0].children[1][0]}")
        
        # the destrination node is yet to be found 
        if(len(selectTree[src].children) > 0):      # if a children array for a selected node is not empty, search it in an attempt to locate dest
            # select a child node from the children array
            for child in selectTree[src].children:
                # recursively call findShortestPath which loops over each cchild node until dest has been founf
                findNode = self.findShortestPath(root, child[0], dest)
                # if findNode has been assigned an int, this means dest has been found, hence the recursive sequence stops
                if not findNode is None:
                    return findNode
                
    
    # function with the ability to store the currently generated graph in a .edgelist file
    def createStoreEdgList(self, graphNum: str):
        fileName = 'graph' + graphNum + '.edgelist'
        lines = ""

        for edge in self.addedEdges:
            x = edge[0]
            y = edge[1]
            tupleIndex = [s[0] for s in self.graph[x]].index(y)
            weight = self.graph[x][tupleIndex][1]
            lines += str(x) + " " + str(y) + " " + str(weight) + "\n"

        with open('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/subsetGraphs/'+fileName, 'w') as fileEdgeList:
            fileEdgeList.writelines(lines)
            print(fileEdgeList, ' created')


