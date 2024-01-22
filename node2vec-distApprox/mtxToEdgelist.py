import os
import random

# this file is used to convert facebook matrix data ('.mtx' file) into a '.edgelist' file which can be interpreted by node2vec and networkx

data_files = os.listdir('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/')


for file in data_files:
    # find '.mtx' file
    if file.endswith('.mtx'):
        fileName = file.replace('.mtx', '')
        fileEdgeList = fileName + 'weighted' + '.edgelist'
        if not fileEdgeList in data_files:
            lines = None
            with open('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/'+file) as fileMtx:
                lines = fileMtx.readlines()
            # convert '.mtx' file into '.edgelist' file
            print(lines[len(lines)-1])
            for i in range(2, len(lines)):
                lines[i] = lines[i].rstrip('\n') + " " + str(random.randint(1,10)) + "\n"
            with open('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/'+fileEdgeList, 'w') as fileEdgeList:
                fileEdgeList.writelines(lines[2:])
                print(fileEdgeList, ' created')