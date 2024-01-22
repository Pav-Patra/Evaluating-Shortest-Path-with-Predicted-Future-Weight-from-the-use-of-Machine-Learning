from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk   # to generate a graph embedding


# defined function takes in a graph drawn using networkx and performs a DeepWalk to generate an embedding
def getEmbedding(myGraph):
    deepwalk = DeepWalk(dimensions=2)    # 2Dimensional Vector Embedding

    deepwalk.fit(myGraph)          # Perform DeepWalks on Networkx graph

    embedding = deepwalk.get_embedding()        # calcuate the embedding from the DeepWalks
    
    print(embedding)
    return(embedding)