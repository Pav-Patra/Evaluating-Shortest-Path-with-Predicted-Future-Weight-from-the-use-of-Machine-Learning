import os
import os.path as osp
import numpy as np

"""
This file is created by Pav Patra
"""
# get current working directory
cwd = os.getcwd()


emd_map = {}

path = osp.join(cwd, "data", "emb")

embedding_file = os.listdir(path)

for file in embedding_file:
    # find '.emd' file
    if file.endswith('.emd'):
        fileName = file
        with open(path+fileName) as embFile:
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