import os
import numpy as np




emd_map = {}

embedding_file = os.listdir('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/emb')

for file in embedding_file:
    # find '.emd' file
    if file.endswith('.emd'):
        fileName = file
        with open('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/emb/'+fileName) as embFile:
            lines = embFile.readlines()


# line = list(map(float, lines[1].split()))
# key = int(line[0])
# val = line[1:]

# keyValPair = {key: val}

# print(keyValPair)


for line in lines:

    parseLine = list(map(float, line.split()))
    #print(parseLine)
    key = int(parseLine[0])
    val = np.array(parseLine[1:])

    if len(val) > 1:
        emd_map[key] = val

print(emd_map)