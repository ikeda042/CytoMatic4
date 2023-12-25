import numpy as np
from numpy.linalg import svd

with open("sk25_btapro_FITC_c_peak_points_xs.txt") as f:
    ys_0 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_0_normalized = []    
    for i in ys_0:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_0_normalized.append(i.tolist())

with open("sk25_bta_FITC_20_peak_points_ys.txt") as f:
    ys_1 = [[float(x.replace("\n","")) for x in line.split(",")] for line in f.readlines()]
    ys_1_normalized = []
    for i in ys_1:
        i = np.array(i)
        i = (i - i.min()) / (i.max() - i.min())
        ys_1_normalized.append(i.tolist())

ys_0_normalized = ys_0_normalized[0:5]
ys_1_normalized = ys_1_normalized[20:25]

import matplotlib.pyplot as plt
import numpy as np

data1 = np.array(ys_0_normalized)
data2 = np.array(ys_1_normalized)
A = np.concatenate([data1,data2],axis=0)
U, S, V_T = svd(A)
print("左特異値ベクトル行列")
print(U)
print("Σ")
print(np.diag(S))
print("右特異値ベクトル行列")
print(V_T)

A = np.concatenate([data1, data2], axis=0)

# Performing Singular Value Decomposition (SVD)
U, S, V_T = svd(A)

# Projecting the data onto the first two singular vectors
reduced_data = np.dot(A, V_T.T[:, :2])

# Separating the projected data for the two original datasets
reduced_data1 = reduced_data[:len(data1)]
reduced_data2 = reduced_data[len(data1):]

import seaborn as sns
sns.set()
# Plotting the results
plt.figure(figsize=(6, 6))
plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], color='blue', label='Negative Ctrl.')
plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], color='red', label='Positive Ctrl.')
plt.title('2D Projection using SVD')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.savefig("svd.png",dpi = 500)
plt.close()
