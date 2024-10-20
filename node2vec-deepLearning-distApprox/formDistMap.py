import networkx as nx 
import numpy as np 
import time
import pickle
import tqdm.auto as tqdm
import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
from sklearn.model_selection import train_test_split
import torch
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

"""
This is a heavily modified file by Pav Patra taken from the following article: 
https://towardsdatascience.com/shortest-path-distance-with-deep-learning-311e19d97569 which takes 
concepts from the paper node2vec: Scalable Feature Learning for Networks by Grover, Aditya and Leskovec, 
Jure (https://doi.org/10.1145/2939672.2939754). The original file contained several issues and 
compatibility issues with my own system.
"""


# This file selects a certain number of nodes in the graph as landmarks and compute their distances from all the rest of the nodes.
# the results are stored as a dictionary into the following file in the 'outputs' dir: distance_map_web-webbase-{time}.pickle
# 'distance_map' dict holds the distance of every node from a given landmark (as a key)

# This file essentially forms the dataset which the model will train from 

# get current working directory
cwd = os.getcwd()

np.random.seed(999)
#edgelist_path = 'C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/socfb-American75weighted.edgelist'
edgelist_path = osp.join(cwd, "data", "socfb-American75weighted.edgelist")
#graph = nx.read_edgelist(edgelist_path, nodetype=int)
graph = nx.read_weighted_edgelist(edgelist_path, nodetype=int)

nodes = list(graph.nodes)
landmarks = np.random.randint(1, len(nodes), 150)    # number of landmarks << number of nodes in graph 

distance_map = {}
distances = np.zeros((len(nodes), ))

for landmark in tqdm.tqdm(landmarks):
    distances[:] = np.inf
    #node_dists = nx.shortest_path_length(graph, landmark)
    node_dists = nx.shortest_path_length(graph, landmark, weight='weight')
    for key, value, in node_dists.items():
        distances[key-1] = value    # since node labels start from 1
        #print(f"For landmark = {landmark} between key = {key-1} distance value = {value}")
    distance_map[landmark] = distances.copy()    # copy as array is reinitialised on loop start

#savePath = 'C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/outputs/distance_map_'+'socfb-American75weighted'+'_'+str(time.time())+'.pickle'
savePath = osp.join(cwd, "outputs", 'distance_map_'+'socfb-American75weighted'+'_'+str(time.time())+'.pickle')
pickle.dump(distance_map, open(savePath, 'wb'))
print('distance_map saved at ', savePath)

print(distance_map)

"""""
for each distance get the corresponding node embedings and combine by averaging them 
"""

# end_map is a dictionary which holds each node as key and it's embedding as value
# read data from data/emb/web-webbase-2001.edgelist

emd_map = {}

#embedding_file = os.listdir('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/emb')
fileName = 'socfb-American75weighted.emd'

openFile = osp.join(cwd, "data", "emb", fileName)

with open(openFile) as embFile:
    lines = embFile.readlines()
    
for line in lines:

    parseLine = list(map(float, line.split()))
    #print(parseLine)
    key = int(parseLine[0])
    val = np.array(parseLine[1:])

    if len(val) > 1:
        emd_map[key] = val


emd_dist_pair = []
for landmark in tqdm.tqdm(list(distance_map.keys())):
    node_distances = distance_map[landmark]
    emd_dist_pair.extend([((emd_map[node]+emd_map[landmark])/2, distance) for node, distance in enumerate(node_distances, 1) if node != landmark and distance != np.inf])

print('length of embedding-distance pairs', len(emd_dist_pair))

# Form numpy ndarrays from embedding-distance dict

x = np.zeros((len(emd_dist_pair), len(emd_dist_pair[0][0])))
y = np.zeros((len(emd_dist_pair),))

for i, tup in enumerate(tqdm.tqdm(emd_dist_pair)):
    x[i] = tup[0]
    y[i] = tup[1]
x = x.astype('float32')             # reduce memory usage
y = y.astype('int')                 # reduce memory isage
print("\nShape of x={} and y={}".format(x.shape, y.shape))
print('size of x={} MB and y={} MB'.format(sys.getsizeof(x)/1024/1024, sys.getsizeof(y)/1024/1024))



# stratify the train/test split using x and y
seedRandom = 9999
np.random.seed(seedRandom)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seedRandom)

print(f"Length of y is {len(y)}")
print(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=seedRandom, shuffle=True, stratify=y)   # issue here
# x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=seedRandom, shuffle=True, stratify=y_train)  # issue here 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=seedRandom, shuffle=True)   # issue here
x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=seedRandom, shuffle=True)  # issue here 

print('shapes of train, validation, test data ', x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape)

# Normalize the data

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_cv = scaler.transform(x_cv)
x_test = scaler.transform(x_test)

trainPath = osp.join(cwd, "outputs", "train_xy_no_sampling_stdScale.pk")
cvPath = osp.join(cwd, "outputs", "val_xy_no_sampling_stdScale.pk")
testPath = osp.join(cwd, "outputs", "test_xy_no_sampling_stdScale.pk")

# saving the split data mean preprocessing can be done once only
pickle.dump((x_train, y_train), open(trainPath, 'wb'))
pickle.dump((x_cv, y_cv), open(cvPath, 'wb'))
pickle.dump((x_test, y_test), open(testPath, 'wb'))