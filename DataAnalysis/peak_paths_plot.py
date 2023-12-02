with open("test_database_peak_points_xs.txt", "r") as f:
    xs = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
with open("test_database_peak_points_ys.txt", "r") as f:
    ys = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

for i,j in zip(xs,ys):
    print(len(i),len(j))
fig = plt.figure(figsize=(6, 6))
n = 0
for i,j in zip(xs,ys):
    #normalize i in range 0 to 1
    i = np.array(i)
    i = (i - i.min()) / (i.max() - i.min())
    #normalize j in range 0 to 1
    j = np.array(j)
    j = (j - j.min()) / (j.max() - j.min())
    plt.scatter(i,j,s= 1,color = "red",marker="o")
    plt.plot(i,j,color = "red",linewidth = 0.1)
    n += 1
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Path Finder Algorithm')
plt.legend()
fig.savefig(f"test.png",dpi = 300)
    