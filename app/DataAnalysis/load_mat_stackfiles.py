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

#reconstruct contour
fig = plt.figure(figsize=[7, 7])
ax = fig.add_subplot(111)
ax.set_aspect("equal")
ax.scatter([i[0] for i in contour], [i[1] for i in contour], s=50, color="lime")
fig.savefig("result.png", dpi=500)





