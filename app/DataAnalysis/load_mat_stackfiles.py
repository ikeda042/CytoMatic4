import scipy.io as sio
import numpy as np
from typing import Any, cast


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
print(len(cast(np.ndarray, cell_i)))
print(cell_i)
