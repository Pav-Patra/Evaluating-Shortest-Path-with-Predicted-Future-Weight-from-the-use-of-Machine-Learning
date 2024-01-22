import Incremental_Shortest_Path as ISP     # my own defined Graph object for the ISP algorithm
import random


graphNum = "2"
graphCount = 1

# Create a graph 

g = ISP.Graph()

g.generateGraph(20)

g.createStoreEdgList(graphNum)

for i in range(5):
    ranNum = random.randint(1, 20)
    for i in range(ranNum):
        g.changeGraph()

    g.createStoreEdgList(graphNum + '_' + str(graphCount))

    graphCount+=1


    