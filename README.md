# Project Documentation 

This file details each of the project's code files and how to execute them.

## Virtual Environment and Python Packages

To execute any of the scripts in this directory, all the relevant packlages must be installed from 'requirements.txt' if not done so alreafy.

One way this can be achieved is by creating a virtual-environment in Python running Python version 3.10.7. This can be done by following these steps.:

1) To begin with, ensure Python 3.10.7 is installed onto your system and set as the system language.
2) In the 'ProjectCodeSubmission directory, run the following command in the terminal to create a new virtual environment: 'python -m venv venv' This creates a new virtualenvironment is created.
3) Activate the virtual environment with the following command: '.\venv\Scripts\activate'(Windows) or 'source venv/bin/activate'(macOS/Linux).
4) Ensure the VSCode Python Extension is installed on your VSCode application.
5) Configure VSCode to use the Virtual Environment. When selecting a Python file in the 'ProjectCodeSubmission' directory, select the Pythonn iterpreter on the bottom left hand corner of the status bar in VSCode. From here, select './venv/Scripts/python.exe' (Windows) or './venv/bin/python' (macOS/Linux)
6) Ensure pip is installed in the virtual environment first by running: 'python -m pip install -U --force pip'
7) Run the following command to install the required packages for this project: 'pip install -r requirements.txt'

TroubleShooting. 'python -m pip install -U --force pip' may encounter issues installing all packages. The remaining packages can be installed using the following commands:
- pip install numpy
- pip install networkx
- pip install matplotlib
- pip install scipy
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- pip install  keyboard
- pip install torch_geometric
- pip install utils
- pip install imblearn
- pip install mlxtend
- pip install torchsummary

## File and Directory Structuring

The following is a directory tree containing all relevant files in this project directory:

- Project Code Submission
    - node2vec-deepLearning-distApprox 
        - data
        - node2vec
        - outputs
        - formDistMap.py
        - mtxToEdgelist.py
        - readEmbedding.py
        - train.py
    - outputs
        - model_best.pth
    - createGraph.py
    - dt_1d_net.py
    - graph_dataset.py
    - graph_models.py
    - graph_train.py
    - Incremental_Shortest_Path.py
    - README.md
    - requirements.txt
    - SP_NN_model.py
    - test_graph.py

Description of each file and directory:

Project Code Submission - the directory holding all of the project's content. This is initially zipped as a zip file.

node2vec-deeplearning-distApprox - directory containing all sub-directories and python files for first machin learning attempt of a shortest path solving neural netwoork. This is a feedforward neural network that utilises the node2vec embedding algorithm.

data - directory that holds the dataset files used to train the feedforward neural network in train.py

node2vec - directory containing the pre-built Python files for the node2vec algorith. This was taken from the following paper's GitHub repository: node2vec: Scalable Feature Learning for Networks by Aditya Grover and Jure Leskovec.

node2vec-deeplearning-distApprox > outputs - directory containing the training pairs of a graph emebdding instance generated using node2vec.

formDistMap.py - This file selects a certain number of nodes in the graph as landmarks and compute their distances from all the rest of the nodes. The results are stored as a dictionary into the following file in the 'outputs' dir: distance_map_web-webbase-{time}.pickle 'distance_map' dict holds the distance of every node from a given landmark (as a key) This file essentially forms the dataset which the model will train from. This is a heavily modified file by Pav Patra taken from the following article: https://towardsdatascience.com/shortest-path-distance-with-deep-learning-311e19d97569 which takes concepts from the paper node2vec: Scalable Feature Learning for Networks by Grover, Aditya and Leskovec, Jure (https://doi.org/10.1145/2939672.2939754). The original file contained several issues and compatibility issues with my own system

mtxToEdgelist.py - Python file converting a .mtx instance of a graph into .edgelist format.

readEmbedding.py - Python file used how to test how to read an embedding file produiced by node2vec.

train.py - Python file containing code to produce the FeedForward Neural Network. This is a heavily modified file by Pav Patra taken from the following article: https://towardsdatascience.com/shortest-path-distance-with-deep-learning-311e19d97569 which takes concepts from the paper node2vec: Scalable Feature Learning for Networks by Grover, Aditya and Leskovec, Jure (https://doi.org/10.1145/2939672.2939754). The original file contained several issues and compatibility issues with my own system.

createGraph.py - this file demonstrates how the Incremental Shortest Path algorithm alongside its Shortest Path Nerual Network used to return the shortest path between any 2 nodes for dynamic, undirected, weighted graphs in real time.

dt_1d_net.py - The methods in this file are utilised by graph_models.py. This file was modified by Sean McLeish: https://github.com/mcleish7/DT-GNN

graph_dataset.py - This Python file contains the methods that randomly generates the dataset that is used to train the DeepThinking Neural Network using graph_train.py. This file was modified by Sean McLeish: https://github.com/mcleish7/DT-GNN and utilised by Pav Patra in custom SP_NN_model.py class

graph_models.py - This Python file contains the defined methods for the DeepThinking Graph Neural Network. This file was modified by Sean McLeish: https://github.com/mcleish7/DT-GNN and utilised by Pav Patra in custom SP_NN_model.py class

graph_train.py - This Python file trains the best case Deep Thinking Graph Neural Network. to reproduce a best case mode, run: 'python graph_train.py --exp=DT_sp --logdir=shortestpath --train --num_steps=20 --dataset=shortestpath --cuda --infinite --dt --lr 0.0001 --prog --alpha=0.2 --json_name dt --rank=16 --gen_rank=0 --vary --random_noise'. This file was modified by Sean McLeish: https://github.com/mcleish7/DT-GNN and utilised by Pav Patra in custom SP_NN_model.py class.

SP_NN_model.py - This Python file utilises the Deep Thinking Graph Neural Network best case model. It takes in an undirected, weighted graph instance and 2 nodes to determine whether edge weight changes have occurred. This is used alongside in conjunction Incremental_Shortest_Path.py in createGraph.py and graph_train.py whilst executing the ISP algorithm to determine when to execute the update method. This file is created by Pav Patra
 
Incremental_Shortest_Path.py - This Python file contains the Incremental Shortest Path algoreithm class, (Graph). The Graph class can be used to generate a dynamic, undirected, weighted graph. The incrementalShortestPath() method in Graph calculates and stores all shortest path distances in the Graph class's data structure. This file is created by Pav Patra.
    
README.md - The current document. 
     
requirements.txt - This file contains all of the required packages and versions to install in the Python virtual environment whilst executing this project's scripts.
    
     
test_graph.py - This Python file is the testing suite for this project. Testing is maintained for both Incremental_Shortest_Path and SP_NN_Model Python files. This file also contains the benchmark testing results of the developed ISP algorithm against Dijkstra's and the Bellman-Ford algorithm. To run this file, simply execute: python test_graph.py. This file is created by Pav Patra

## The Incremental Shortest Path Algorithm + Graph Neural Network

The Incremental Shortest Path algorithm consists of two main classes:

- The 'Graph' class located in Incremental_Shortest_Path.py
- The 'SP_NN' class located in SP_NN_model.py.

Both files are created by myself.

Both of these classes are used in conjunction in the createGraph.py and test_graph.py Python files. createGraph.py demonstrates the execution of the Incremental Shortest Path algorithm on a dynamic, undirected, weighted graph with a specified number of nodes. 

Running createGraph.py:

1) Run the following command in the ProjectCodeSubmission directory: 'python createGraph.py'
2) Specify the number of nodes after the first prompt. For best results, keep below 150 nodes.
3) Fetch an intial shortest path between any 2 nodes by sepcifying the node numbers in the next 2 prompts
4) The graph is now in dynamic mode where continous edge weights are changin repeatedly
5) Hold space to exit dynamic mode
6) Follow the next two prompts to fetch a nmew shortest path
7) Press (1) to continue execution for another iteration or (2) to exit the algorithm

testGraph.py is the testing suite used alongside the engineering of the entire Incremental Shortest Path algorithm. It contains several methods that test differnt modules created in Incremental_Shortest_Path.py and SP_NN_model.py. Most importantly, it contains the benchmark testing of the Incremental Shortest Path algforith along with its Graph Neural Network against Dijkstra's algorithm, the Bellman-Ford algorithm and the Incremental Shortest Path algorithm without the Graph Neural Network implementation from SP_NN. Important note: The performance of the ISP algorith varies between system specs.

Running test_graph.py:

1) Run the following command in the ProjectCodeSubmission directory: 'python test_graph.py'
2) Wait for execution to complete
3) Benchmark results are displayed in a separate window as three separate graphs. 

## Initial Node2Vec FeedFowrard Neural Network Attempt

By entering the node2vec-deeplearning directory, you are able to run the train.py Python file using the following command:

'python train.py'

This command creates and executes the feedforwards neural network. A best model is created that is capable of approximating the shortest path in a graph. The 'socfb-American75weighted.edgelist' dataset is used to train this model. This dataset is an undirected weighted graph containing 217,662 edges and installed from: "http://networkrepository.com". 

The final results displayed are the accuracy scores as two separate graphs as well as the Mean Square Error(MSE) and Mean Absolute Error(MAE) scores. These display the inaccurate scores produced by using this intial attempt of a machine learning implementation. This resulted in a change of direction towards Graph Neural Networks as seen above.


