import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import sqrt

#TSP_FILE = "/u/mnappo/git/aco-tsp/data/dj38.tsp"
#SOLVE_FILE = "/u/mnappo/dj38.sol"

TSP_FILE = "/u/mnappo/git/aco-tsp/data/ts10.tsp"
SOLVE_FILE = "/u/mnappo/ts10.sol"

# calc weight from e[0] to e[1]
def weight(e, x, y):
    src_x, src_y = x[e[0]], y[e[0]]
    dst_x, dst_y = x[e[1]], y[e[1]]
    d = sqrt((src_x-dst_x)**2 + (src_y-dst_y)**2)
    return d

# Read data from file
with open(TSP_FILE) as f:
    lines = f.read().splitlines()
    i = 0
    for line in lines:
        if line == "NODE_COORD_SECTION":
            i += 1
            break
        i += 1
    data = [[float(x) for x in l.split()] for l in lines[i:]]

labels, x, y = map(list, zip(*data))
n = len(data)
print(n)
A = [[0.0 for i in range(n)] for i in range(n)]
for i in range(n):
    for j in range(n):
        A[i][j] = weight((i,j), x, y)
A = np.array(A)
#print(A)

# calc path length
def path_len(path):
    l = 0.0
    for i in range(1, len(path)):
        e = (path[i-1], path[i])
        l += weight(e, x, y)
    return l

print(path_len([0,8,1,4,6,5,7,9,2,3]))
print(path_len([0,9,5,2,7,3,7,6,4,1, 2, 1, 9]))
print(path_len([0,9,1,7,6,6,7,8,7,1]))

def pathlen_from_file(file):
    with open(file) as f:
        path = [int(x) for x in ' '.join(f.readlines()[1:]).split()]
    return path_len(path)

# get pathlen of a path string in the form [1 2 3 4 5]
def pathlen_from_str(path_str):
    path = [int(x) for x in path_str[1:len(path_str)-1].split()]
    return path_len(path)

print("solve", pathlen_from_file(SOLVE_FILE))
print("ours", pathlen_from_str("[ 0 9 5 2 7 3 8 6 4 1 ]"))
#print("ours (1k ants, 1k iters)", pathlen_from_str("[ 0 37 27 4 15 33 21 35 22 26 31 1 14 32 34 8 11 17 13 23 36 9 12 2 24 16 30 25 28 20 5 7 3 18 6 29 19 10 ]"))
#print("ours (10 ants, 10 iter)", pathlen_from_str("[ 0 36 35 31 26 27 8 13 32 20 12 18 33 11 2 37 34 10 29 14 21 1 19 4 28 9 7 24 5 30 6 25 15 3 22 23 16 17 ]"))

'''
# display points
plt.scatter(x,y)
for i, l in enumerate(labels):
    plt.annotate(int(l), (x[i], y[i]))
plt.show()
'''

G = nx.from_numpy_matrix(A)
nx.draw(G)
plt.show()

