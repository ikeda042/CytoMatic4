import scipy.io as sio
import numpy as np
from typing import Any, cast
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


class Cell:
    def __init__(self, file_name) -> None:
        self.mat_data: Any = sio.loadmat(file_name=file_name)
        self.keys = self.mat_data.keys()

    def get_cellList(self) -> np.ndarray:
        return self.mat_data["cellList"]


cell = Cell("Ph_com_mesh_signal.mat")

data_i: np.ndarray = cell.get_cellList()
cell_i: np.ndarray | None = None
for i in range(11):
    data_i = data_i[0]
    if i == 8:
        cell_i = data_i
cell_i = cast(np.ndarray, cell_i)
print(len(cell_i))
contour = cell_i[2]
print(contour)

# reconstruct contour
fig = plt.figure(figsize=[7, 7])
ax = fig.add_subplot(111)
ax.set_aspect("equal")
ax.scatter([i[0] for i in contour], [i[1] for i in contour], s=50, color="lime")
fig.savefig("result_contour.png", dpi=500)
plt.close()


"""
@params_dict
0: model
1: algorithm
2: birthframe
3: mesh
4: stage
5: polarity
6: timelapse
7: box
8: divisions
9: ancestors
10: descendants
11: signal0
12: signal2
13: steplength
14: length
15: lengthvector
16: area
17: steparea
18: stepvolume
19: volume
"""

params_dict = {
    "model": 0,
    "algorithm": 1,
    "birthframe": 2,
    "mesh": 3,
    "stage": 4,
    "polarity": 5,
    "timelapse": 6,
    "box": 7,
    "divisions": 8,
    "ancestors": 9,
    "descendants": 10,
    "signal0": 11,
    "signal2": 12,
    "steplength": 13,
    "length": 14,
    "lengthvector": 15,
    "area": 16,
    "steparea": 17,
    "stepvolume": 18,
    "volume": 19,
}

cells = cell.get_cellList()[0][0][0][0][0][0]

cell_num = len(cells)

for cell_id in range(cell_num):
    cell_0_mesh = cell.get_cellList()[0][0][0][0][0][0][cell_id][0][0][params_dict["mesh"]]
    print(cell_0_mesh[0])
    print(len(cell_0_mesh[0]))

    # reconstruct mesh
    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    for i in cell_0_mesh:
        ax.plot([i[0], i[2]], [i[1], i[3]], color="r")
    fig.savefig("result_mesh.png", dpi=500)
    plt.close()

print("+++++++++++++++test field+++++++++++++++")
cell_id = 0
print(cell.get_cellList()[0][0][0][0][0][0][cell_id][0][0][params_dict["mesh"]])