from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from numpy.linalg import inv
from .combine_images import combine_images_function
from scipy.integrate import quad
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from .components import create_dirs, basis_conversion, calc_arc_length
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

def polynomial_regression(U, k, min_u1, max_u1, u2_c, u1_contour, u2_contour) -> float:
    plt.scatter(u1_contour, u2_contour, s=5, color="lime")
    W = np.array([[i**j for j in range(k, -1, -1)] for i in [i[1] for i in U]])
    f = np.array([i[0] for i in U])
    theta = inv(W.T @ W) @ W.T @ f

    x = np.linspace(min_u1, max_u1, 1000)
    y_pred = sum([theta[j] * x ** (k - j) for j in range(k + 1)])

    def arc_length_integrand(u1):
        dydu1 = sum([theta[j] * (k - j) * u1 ** (k - j - 1) for j in range(k + 1)])
        return np.sqrt(1 + dydu1**2)

    length, _ = quad(arc_length_integrand, min_u1, max_u1)
    # plt.plot(x, y_pred, color="blue", linewidth=1)
    # plt.scatter(min_u1, sum([theta[j] * min_u1**(k-j) for j in range(k+1)]), s=100, color="red", zorder=100, marker="x")
    # plt.scatter(max_u1, sum([theta[j] * max_u1**(k-j) for j in range(k+1)]), s=100, color="red", zorder=100, marker="x")
    # plt.xlim(min_u1 - 40, max_u1 + 40)
    # plt.ylim(u2_c - 40, u2_c + 40)
    # plt.xlabel("u1")
    # plt.ylabel("u2")
    # plt.title(f"Polynomial Regression with k={k}")
    # plt.axis("equal")
    # plt.grid()
    # #plt text of length
    # plt.text(min_u1+50, u2_c+25, f"L={round(length*0.0625,2)}(um)", color="red", ha="center", va="top")
    # plt.savefig(f"poly_reg_k{k}.png")
    # plt.close()
    return length

def combine_images_function_singlemode(total_rows, total_cols, image_size, num_images, filename, single_layer_mode, dual_layer_mode):
    result_image = np.zeros((total_rows * image_size, total_cols * image_size, 3), dtype=np.uint8)
    num_images += 1
    print("=======================================================")
    print(image_size)
    for i in range(total_rows):
        for j in range(total_cols):
            image_index = i * total_cols + j 
            if image_index <num_images:
                image_path = f'Cell/ph/{image_index}.png'  
                print(image_path)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (image_size, image_size))
                result_image[i * image_size: (i + 1) * image_size,
                            j * image_size: (j + 1) * image_size] = img
    plt.axis('off')
    cv2.imwrite(f'{filename}_ph.png', result_image)
    if not single_layer_mode:
        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j  
                if image_index <num_images:
                    image_path = f'Cell/replot/{image_index}.png' 
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[i * image_size: (i + 1) * image_size,
                                j * image_size: (j + 1) * image_size] = img
        plt.axis('off')
        cv2.imwrite(f'{filename}_replot.png', result_image)



Base = declarative_base()

class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB)
    # img_fluo2 = Column(BLOB)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)

def data_analysis_singlemode(
    db_name: str = "test.db",
    image_size: int = 100,
    out_name: str = "cell",
    dual_layer_mode: bool = True,
    single_layer_mode: bool = False,
):
       ##############################################################
    n = -1
    cell_lengths = []
    agg_tracker = 0
    means = []
    meds = []
    agg_bool = []
    vars = []
    max_intensities = []
    max_int_minus_med = []
    mean_fluo_raw_intensities = []
    skewnesses = []
    kurtosises = []

    """
    二重染色用データ
    """
    mean_fluo_raw_intensities_2 = []

    """
    テクスチャ解析パラメータ
    """
    energies = []
    contrasts = []
    dice_similarities = []
    homogeneities = []
    correlations = []
    ASMs = []
    smoothnesses = []

    """
    ヒストグラム解析パラメータ
    """
    cumulative_frequencys = []

    """
    投影データ
    """
    projected_points_xs = []
    projected_points_ys = []
    peak_points: list[list[float]] = []

    """
    輝度の密度関数(split 面積ベース)
    """
    sum_brightness = []
    sum_brightnesses = []
    ##############################################################

    create_dirs(
        [
            "RealTimeData",
            "Cell",
            "Cell/ph",
            "Cell/fluo1",
            "Cell/fluo2",
            "Cell/histo",
            "Cell/histo_cumulative",
            "Cell/replot",
            "Cell/replot_map",
            "Cell/fluo1_incide_cell_only",
            "Cell/fluo2_incide_cell_only",
            "Cell/gradient_magnitudes",
            "Cell/GLCM",
            "Cell/unified_cells",
            "Cell/3dplot",
            "Cell/projected_points",
            "Cell/peak_path",
            "Cell/sum_brightness",
            "Cell/gradient_magnitude_replot"
        ]
    )
    sns.set()
    plt.style.use("dark_background")
    engine = create_engine(f"sqlite:///{db_name}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        cells = session.query(Cell).all()
        for cell in tqdm(cells):
            if cell.manual_label != "N/A" and cell.manual_label != None:
                
                print("###############################################")
                print(cell.cell_id)
                print("###############################################")
                n += 1
                image_ph = cv2.imdecode(
                        np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                cv2.imwrite(
                            f"RealTimeData/ph.png", image_ph
                        )
                image_size = image_ph.shape[0]
                image_ph_copy = image_ph.copy()
                position = (0, 15)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                thickness = 1
                cv2.putText(
                    image_ph,
                    f"{cell.cell_id}",
                    position,
                    font,
                    font_scale,
                    font_color,
                    thickness,
                )
                cv2.imwrite(f"Cell/ph/{n}.png", image_ph_copy)


                coords_inside_cell_1 = []
                for i in range(image_size):
                    for j in range(image_size):
                        if (
                            cv2.pointPolygonTest(
                                pickle.loads(cell.contour), (i, j), False
                            )
                            >= 0
                        ):
                            coords_inside_cell_1.append([i, j])


                cell_contour = [list(i[0]) for i in pickle.loads(cell.contour)]
                print(cell_contour)
                contour = [
                        [j, i] for i, j in [i[0] for i in pickle.loads(cell.contour)]
                    ]
                X = np.array(
                     [
                            [i[1] for i in coords_inside_cell_1],
                            [i[0] for i in coords_inside_cell_1],
                        ]
                )
                (
                        u1,
                        u2,
                        u1_contour,
                        u2_contour,
                        min_u1,
                        max_u1,
                        u1_c,
                        u2_c,
                        U,
                        contour_U,
                    ) = basis_conversion(
                        contour, X, cell.center_x, cell.center_y, coords_inside_cell_1
                    )
                min_u1, max_u1 = min(u1), max(u1)
                fig = plt.figure(figsize=[6, 6])
                cmap = plt.get_cmap("inferno")
                x = np.linspace(0, 100, 1000)
                W = np.array(
                        [[i**4, i**3, i**2, i, 1] for i in [i[1] for i in U]]
                    )
                print(W)
                f = np.array([i[0] for i in U])
                theta = inv(W.transpose() @ W) @ W.transpose() @ f
                x = np.linspace(min_u1, max_u1, 1000)
                y = [
                        theta[0] * i**4
                        + theta[1] * i**3
                        + theta[2] * i**2
                        + theta[3] * i
                        + theta[4]
                        for i in x
                    ]
                cell_length = calc_arc_length(theta, min_u1, max_u1)
                print(cell_lengths)
                cell_lengths.append([cell.cell_id, cell_length])
                plt.plot(x, y, color="blue", linewidth=1)
                plt.scatter(
                    min_u1,
                    theta[0] * min_u1**4
                    + theta[1] * min_u1**3
                    + theta[2] * min_u1**2
                    + theta[3] * min_u1
                    + theta[4],
                    s=100,
                    color="red",
                    zorder=100,
                    marker="x",
                )
                plt.scatter(
                    max_u1,
                    theta[0] * max_u1**4
                    + theta[1] * max_u1**3
                    + theta[2] * max_u1**2
                    + theta[3] * max_u1
                    + theta[4],
                    s=100,
                    color="red",
                    zorder=100,
                    marker="x",
                )

                plt.scatter(u1_contour, u2_contour, s=5, color="lime")
                plt.xlabel("u1")
                plt.ylabel("u2")
                plt.axis("equal")
                plt.xlim(min_u1 - 80, max_u1 + 80)
                plt.ylim(u2_c - 80, u2_c + 80)
                fig.savefig(f"Cell/replot/{n}.png")
                fig.savefig(f"RealTimeData/replot.png")
                plt.close()

    total_rows = int(np.sqrt(n))+ 1
    total_cols = n//total_rows + 1
    num_images = n
    combine_images_function_singlemode(total_rows, total_cols, image_size, num_images, out_name,single_layer_mode, dual_layer_mode) 


