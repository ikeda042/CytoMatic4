import scipy.io as sio
import numpy as np
from typing import Any, cast
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
import cv2

dir_name = "Matlab"
if os.path.exists(dir_name):
    shutil.rmtree(dir_name)
os.makedirs(dir_name)
os.mkdir("Matlab/contours")
os.mkdir("Matlab/meshes")
os.mkdir("Matlab/overlay")

matplotlib.use("Agg")
plt.style.use("dark_background")


class CellMat:
    def __init__(self, file_name) -> None:
        self.file_name: str = file_name
        self.mat_data: Any = sio.loadmat(file_name=file_name)
        self.keys: list[str] = self.mat_data.keys()
        self.params_dict: dict = {
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
        self.cell_list: list[np.ndarray] = self.mat_data["cellList"]
        self.meshes: list[np.ndarray] = []
        self.contours: list[np.ndarray] = []

    def extract_contours(self) -> None:
        cell_id = 0
        cells = self.cell_list[0][0][0][0][0][0]
        for cell_id in range(len(cells) - 1):
            cell_i_contour = cells[cell_id][0][0][self.params_dict["birthframe"]]
            print(cell_i_contour)
            try:
                # reconstruct contour
                fig = plt.figure(figsize=[7, 7])
                ax = fig.add_subplot(111)
                ax.set_aspect("equal")
                ax.scatter(
                    [i[0] for i in cell_i_contour],
                    [i[1] for i in cell_i_contour],
                    s=50,
                    color="lime",
                )
                fig.savefig(f"Matlab/contours/result_contour_{cell_id}.png", dpi=100)
                plt.close()
                self.contours.append(cell_i_contour)
            except Exception as e:
                print(e)
                print(cell_id)

    def extract_meshes(self) -> None:
        cells = self.cell_list[0][0][0][0][0][0]
        cell_num = len(cells)
        for cell_id in range(cell_num):
            cell_i_mesh = cells[cell_id][0][0][self.params_dict["mesh"]]
            self.meshes.append(cell_i_mesh)
            print(cell_i_mesh[0])
            print(len(cell_i_mesh[0]))
            # reconstruct mesh
            fig = plt.figure(figsize=[7, 7])
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")
            for i in cell_i_mesh:
                ax.plot([i[2], i[0]], [i[3], i[1]], color="lime")
            fig.savefig(f"Matlab/meshes/result_mesh_{cell_id}.png", dpi=100)
            plt.close()

    def overlay_meshes(self) -> None:
        for i in range(len(self.contours)):
            fig = plt.figure(figsize=[7, 7])
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")
            ax.scatter(
                [i[0] for i in self.contours[i]],
                [i[1] for i in self.contours[i]],
                s=50,
                color="lime",
            )
            for j in self.meshes[i]:
                ax.plot([j[2], j[0]], [j[3], j[1]], color="lime")
            fig.savefig(f"Matlab/overlay/overlay_{i}.png", dpi=100)
            plt.close()

    def combine_images(self) -> None:
        image_size = 700
        num_images = len(os.listdir("Matlab/contours")) - 1
        total_rows = int(np.sqrt(num_images)) + 1
        total_cols = num_images // total_rows + 1
        result_image = np.zeros(
            (total_rows * image_size, total_cols * image_size, 3), dtype=np.uint8
        )
        num_images += 1
        print("=======================================================")
        print(image_size)
        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j
                if image_index < num_images:
                    image_path = f"Matlab/contours/result_contour_{image_index}.png"
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[
                        i * image_size : (i + 1) * image_size,
                        j * image_size : (j + 1) * image_size,
                    ] = img
        plt.axis("off")
        cv2.imwrite(f"{self.file_name}_contours.png", result_image)

        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j
                if image_index < num_images:
                    image_path = f"Matlab/meshes/result_mesh_{image_index}.png"
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[
                        i * image_size : (i + 1) * image_size,
                        j * image_size : (j + 1) * image_size,
                    ] = img
        plt.axis("off")
        cv2.imwrite(f"{self.file_name}_meshes.png", result_image)

        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j
                if image_index < num_images:
                    image_path = f"Matlab/overlay/overlay_{image_index}.png"
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[
                        i * image_size : (i + 1) * image_size,
                        j * image_size : (j + 1) * image_size,
                    ] = img
        plt.axis("off")
        cv2.imwrite(f"{self.file_name}_overlay.png", result_image)

    def extract_peak_paths(self) -> None:
        peak_paths = []
        cells = self.cell_list[0][0][0][0][0][0]
        for cell_id in range(len(cells) - 1):
            cell_i_peak_path = cells[cell_id][0][0][self.params_dict["signal2"]]
            peak_paths.append(cell_i_peak_path)
        print(peak_paths)
        with open(f"{self.file_name}_peak_paths.txt", "w") as f:
            for path in peak_paths:
                path = [str(i[0]) for i in path]
                f.write(",".join(path) + "\n")


cell_mat = CellMat("Ph_com_mesh_signal.mat")
cell_mat.extract_meshes()
cell_mat.extract_contours()
cell_mat.overlay_meshes()
cell_mat.combine_images()
cell_mat.extract_peak_paths()
