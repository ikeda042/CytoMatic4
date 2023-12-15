from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import numpy as np 
import cv2 
import pickle
import matplotlib.pyplot as plt 
import sqlite3 
from sqlalchemy import update
from numpy.linalg import eig, inv
import os
from .combine_images import combine_images_function
from scipy.integrate import quad
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew
from .components import create_dirs, calc_gradient, basis_conversion, calc_arc_length
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import numpy as np

def find_minimum_distance_and_point(a, b, c, d, e, x_Q, y_Q):
    # 4次式 f(x) の定義
    def f_x(x):
        return a*x**4 + b*x**3 + c*x**2 + d*x + e

    # 点Qから関数上の点までの距離 D の定義
    def distance(x):
        return np.sqrt((x - x_Q)**2 + (f_x(x) - y_Q)**2)

    # scipyのminimize関数を使用して最短距離を見つける
    # 初期値は0とし、精度は低く設定して計算速度を向上させる
    result = minimize(distance, 0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4})

    # 最短距離とその時の関数上の点
    x_min = result.x[0]
    min_distance = distance(x_min)
    min_point = (x_min, f_x(x_min))

    return min_distance, min_point

Base = declarative_base()
class Cell(Base):
    __tablename__ = 'cells'
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label  = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB) 
    img_fluo1 = Column(BLOB)
    # img_fluo2 = Column(BLOB)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)


#######################################################
# 資料作成用関数
def adjust_brightness(image: np.ndarray, brightness_factor: float) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("Image should be in uint8 format")
    image_float = image.astype(np.float32)
    image_bright = image_float * brightness_factor
    image_bright_clipped = np.clip(image_bright, 0, 255)
    result_image = image_bright_clipped.astype(np.uint8)
    return result_image

def unify_images_ndarray2(image1, image2, image3, output_name):
    combined_width = image1.shape[1] + image2.shape[1] + image3.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0], image3.shape[0])
    
    canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Image 1
    canvas[:image1.shape[0], :image1.shape[1]] = image1

    # Image 2
    offset_x_image2 = image1.shape[1]
    canvas[:image2.shape[0], offset_x_image2:offset_x_image2+image2.shape[1]] = image2

    # Image 3
    offset_x_image3 = offset_x_image2 + image2.shape[1]
    canvas[:image3.shape[0], offset_x_image3:offset_x_image3+image3.shape[1]] = image3

    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def unify_images_ndarray(image1, image2, output_name):
    combined_width = image1.shape[1] + image2.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0])
    
    canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    canvas[:image1.shape[0], :image1.shape[1], :] = image1
    canvas[:image2.shape[0], image1.shape[1]:, :] = image2
    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

def polynomial_regression(U, k, min_u1, max_u1, u2_c,u1_contour,u2_contour) -> float:
    plt.scatter(u1_contour,u2_contour,s = 5,color = "lime" )
    W = np.array([[i**j for j in range(k, -1, -1)] for i in [i[1] for i in U]])
    f = np.array([i[0] for i in U])
    theta = inv(W.T @ W) @ W.T @ f

    x = np.linspace(min_u1, max_u1, 1000)
    y_pred = sum([theta[j] * x**(k-j) for j in range(k+1)])

    def arc_length_integrand(u1):
        dydu1 = sum([theta[j] * (k-j) * u1**(k-j-1) for j in range(k+1)])
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

def data_analysis(db_name:str = "test.db", image_size:int = 100,out_name:str ="cell",dual_layer_mode:bool = True,single_layer_mode:bool = False):
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
    peak_points:list[list[float]] = []
    ##############################################################
    
    create_dirs(["Cell","Cell/ph","Cell/fluo1","Cell/fluo2","Cell/histo","Cell/histo_cumulative","Cell/replot","Cell/replot_map","Cell/fluo1_incide_cell_only","Cell/fluo2_incide_cell_only","Cell/gradient_magnitudes","Cell/GLCM","Cell/unified_cells","Cell/3dplot","Cell/projected_points","Cell/peak_path"])

    engine = create_engine(f'sqlite:///{db_name}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        cells = session.query(Cell).all()
       
        for cell in tqdm(cells):
            if  cell.manual_label != "N/A" and cell.manual_label!= None :
                print("###############################################")
                print(cell.cell_id)
                print("###############################################")
                n+=1

                """
                Load image
                """
                image_ph = cv2.imdecode(np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_COLOR)
                image_ph_copy = image_ph.copy()
                image_size = image_ph.shape[0]
                cv2.drawContours(image_ph_copy,pickle.loads(cell.contour),-1,(0,255,0),1)
                position = (0, 15)  
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5 
                font_color = (255, 255, 255)  
                thickness = 1 
                cv2.putText(image_ph, f"{cell.cell_id}", position, font, font_scale, font_color, thickness)
                cv2.imwrite(f"Cell/ph/{n}.png",image_ph_copy)
                


                cell_contour = [list(i[0]) for i in pickle.loads(cell.contour)]
                print(cell_contour)

                coords_inside_cell_1,  points_inside_cell_1 = [], []
                coords_inside_cell_2,  points_inside_cell_2 = [], []
                if not single_layer_mode:
                    image_fluo1 = cv2.imdecode(np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    fluo_out1 = cv2.imdecode(np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.drawContours(fluo_out1,pickle.loads(cell.contour),-1,(0,255,0),2)
                    cv2.imwrite(f"Cell/fluo1/{n}.png",fluo_out1)
                    output_image =  np.zeros((image_size,image_size),dtype=np.uint8)
                    # cv2.drawContours(output_image, [pickle.loads(cell.contour)], 0, 255, thickness=cv2.FILLED)
                    for i in range(image_size):
                        for j in range(image_size):
                            if cv2.pointPolygonTest(pickle.loads(cell.contour), (j,i), False)>=0:
                                output_image[i][j] = image_fluo1[i][j]
                                
                    cv2.imwrite(f"Cell/fluo1_incide_cell_only/{n}.png",output_image)

                if dual_layer_mode:
                    image_fluo2 = cv2.imdecode(np.frombuffer(cell.img_fluo2, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    fluo_out2 = cv2.imdecode(np.frombuffer(cell.img_fluo2, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.drawContours(fluo_out2,pickle.loads(cell.contour),-1,(0,0,255),1)
                    cv2.imwrite(f"Cell/fluo2/{n}.png",fluo_out2)

                    output_image2 =  np.zeros((image_size,image_size),dtype=np.uint8)
                    # cv2.drawContours(output_image, [pickle.loads(cell.contour)], 0, 255, thickness=cv2.FILLED)

                    for i in range(image_size):
                        for j in range(image_size):
                            if cv2.pointPolygonTest(pickle.loads(cell.contour), (j,i), False)>=0:
                                output_image2[i][j] = image_fluo2[i][j]
                                
                    cv2.imwrite(f"Cell/fluo2_incide_cell_only/{n}.png",output_image2)


                if not single_layer_mode:
                    ############################以下勾配計算##################################
                    # Sobelフィルタを適用してX方向の勾配を計算
                    sobel_x = cv2.Sobel(output_image, cv2.CV_64F, 1, 0, ksize=3)

                    # Sobelフィルタを適用してY方向の勾配を計算
                    sobel_y = cv2.Sobel(output_image, cv2.CV_64F, 0, 1, ksize=3)

                    # 勾配の合成（勾配強度と角度を計算）
                    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

                    # 勾配の強度を正規化
                    # gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                    # 勾配強度画像を保存
                    cv2.imwrite(f'Cell/gradient_magnitudes/gradient_magnitude{n}.png', gradient_magnitude)


                if True:
                    if not single_layer_mode:
                        for i in range(image_size):
                            for j in range(image_size):
                                if cv2.pointPolygonTest(pickle.loads(cell.contour), (i,j), False)>=0:
                                    coords_inside_cell_1.append([i,j])
                                    points_inside_cell_1.append(output_image[j][i])
                    if single_layer_mode:
                        for i in range(image_size):
                            for j in range(image_size):
                                if cv2.pointPolygonTest(pickle.loads(cell.contour), (i,j), False)>=0:
                                    coords_inside_cell_1.append([i,j])
                    if dual_layer_mode:
                        for i in range(image_size):
                            for j in range(image_size):
                                if cv2.pointPolygonTest(pickle.loads(cell.contour), (i,j), False)>=0:
                                    coords_inside_cell_2.append([i,j])
                                    points_inside_cell_2.append(image_fluo2[j][i])
                    # Basis conversion
                    contour = [[j,i] for i,j in [i[0] for i in pickle.loads(cell.contour)]]
                    X = np.array([[i[1] for i in coords_inside_cell_1],[i[0] for i in coords_inside_cell_1]])

                    u1,u2,u1_contour,u2_contour,min_u1,max_u1,u1_c,u2_c,U,contour_U = basis_conversion(contour,X,cell.center_x,cell.center_y,coords_inside_cell_1)
                    min_u1, max_u1 = min(u1), max(u1)
                    fig = plt.figure(figsize=[6,6])
                    cmap = plt.get_cmap('inferno')
                    x = np.linspace(0,100,1000)
                    if not single_layer_mode:
                        max_points = max(points_inside_cell_1)
                        plt.scatter(u1,u2,c =[i/max_points for i in points_inside_cell_1],s = 10,cmap=cmap )
                        plt.scatter(u1_contour,u2_contour,s = 10,color = "lime" )
                        plt.grid()

                    W = np.array([[i**4,i**3,i**2,i,1] for i in [i[1] for i in U]])
                    f = np.array([i[0] for i in U])
                    theta = inv(W.transpose()@W)@W.transpose()@f
                    x = np.linspace(min_u1,max_u1,1000)
                    y = [theta[0]*i**4+theta[1]*i**3 + theta[2]*i**2+theta[3]*i + theta[4] for i in x]

                    cell_length = calc_arc_length(theta,min_u1,max_u1)
                    print(cell_lengths)
                    cell_lengths.append([cell.cell_id,cell_length])

                    plt.plot(x,y,color = "blue",linewidth=1)
                    plt.scatter(min_u1,theta[0]*min_u1**4+theta[1]*min_u1**3 + theta[2]*min_u1**2+theta[3]*min_u1 + theta[4],s = 100,color = "red",zorder = 100,marker = "x")
                    plt.scatter(max_u1,theta[0]*max_u1**4+theta[1]*max_u1**3 + theta[2]*max_u1**2+theta[3]*max_u1 + theta[4],s = 100,color = "red",zorder = 100,marker = "x")
                    plt.xlim(min_u1-80,max_u1+80)
                    plt.ylim(u2_c-80,u2_c+80)
                    plt.xlabel("u1")
                    plt.ylabel("u2")
                    plt.axis("equal")

                    """
                    多項式回帰のKを検討（試験的）
                    """
                    # plt.close()
                    # r2s = []
                    # a, b = 4, 20
                    # if n == 1:
                    #     for i in range(a,b):
                    #         r2_i = polynomial_regression(U, i, min_u1, max_u1, u2_c,u1_contour,u2_contour)
                    #         r2s.append(r2_i)
                    #     fig_testK = plt.figure(figsize=[6,6])
                    #     plt.plot([i for i in range(a,b)],r2s)
                    #     plt.xlabel("K")
                    #     plt.ylabel("R2")
                    #     plt.grid()
                    #     fig_testK.savefig(f"testK.png")

                    normalized_points = [i/max_points for i in points_inside_cell_1]

                    #######################################################統計データ#######################################################
                    med = sorted(normalized_points)[len(normalized_points)//2]
                    med_raw = sorted(points_inside_cell_1)[len(points_inside_cell_1)//2]
                    meds.append(med)
                    means.append(sum(normalized_points)/len(normalized_points))
                    vars.append(np.var(normalized_points))
                    max_intensities.append(max_points)
                    max_int_minus_med.append(max_points-med_raw)
                    mean_fluo_raw_intensities.append(sum(points_inside_cell_1)/len(points_inside_cell_1))
                    if dual_layer_mode:
                        mean_fluo_raw_intensities_2.append(sum(points_inside_cell_2)/len(points_inside_cell_2))
                    #######################################################統計データ#######################################################
                    plt.text(u1_c,u2_c+25,s=f"Mean:{round(sum(normalized_points)/len(normalized_points),3)}\nMed:{round(sorted(normalized_points)[len(normalized_points)//2],3)}\nCell length(μm):{round(cell_length*0.0625,2)}",color = "red",ha="center",va="top")
                    
                    # if med < 0.7:
                    #     plt.scatter(u1_c,u2_c-30,s = 150,color = "red",zorder = 100)
                    #     agg_tracker += 1
                    #     agg_bool.append(1)
                    # else:
                    #     agg_bool.append(0)
                    fig.savefig(f"Cell/replot/{n}.png")
                    fig.savefig(f"replot.png")
                    plt.close()

                    #######################################################ヒストグラム解析#######################################################
                    """
                    正規化した細胞内輝度によるヒストグラムの描画
                    """
                    fig_histo = plt.figure(figsize=[6,6])
                    plt.hist(points_inside_cell_1,bins=100)
                    plt.xlim(0,255)
                    plt.xlabel("Fluo. intensity")
                    plt.ylabel("Frequency")
                    plt.grid()
                    fig_histo.savefig(f"Cell/histo/{n}.png")
                    plt.close()
                    data = [i/255 for i in points_inside_cell_1]
                    skewness = skew(data)
                    kurtosis_ = kurtosis(data)
                    skewnesses.append(skewness)
                    kurtosises.append(kurtosis_)

                    #######################################################ヒストグラム解析（累積頻度）#######################################################

                    fig_histo_cumulative = plt.figure(figsize=[6,6])
                    plt.hist(normalized_points,bins=100,cumulative=True)
                    # 0から255までの頻度を計算
                    frequency = np.bincount(points_inside_cell_1, minlength=256)
                    # 累積頻度の計算
                    cumulative_frequency = np.cumsum(frequency)
                    cumulative_frequency = cumulative_frequency / cumulative_frequency[-1]
                    cumulative_frequencys.append(cumulative_frequency)
                    plt.plot(cumulative_frequency)
                    plt.title('Cumulative Frequency Plot')
                    plt.xlabel('Value (0 to 255)')
                    plt.ylabel('Cumulative Frequency')
                    plt.xlim(-10, 255)
                    plt.ylim(0, 1.05)
                    plt.grid(True)
                    fig_histo_cumulative.savefig(f"Cell/histo_cumulative/{n}.png")
                    plt.close()

                    #######################################################ヒストグラム解析（累積頻度のデルタ）#######################################################
                    fig_histo_cumulative_delta = plt.figure(figsize=[6,6])
                    plt.plot([i for i in range(0,255)],[cumulative_frequency[i+1]-cumulative_frequency[i] for i in range(0,255)])
                    plt.xlabel('Value (0 to 255)')
                    plt.ylabel('Delta Cumulative Frequency')
                    plt.xlim(-1, 255)
                    plt.ylim(0, 0.04)
                    plt.grid(True)
                    fig_histo_cumulative_delta.savefig(f"histo_cumulative_delta.png")
                    plt.close()
                    #######################################################3dプロット
                    fig_3d = plt.figure(figsize=[6,6])
                    ax = fig_3d.add_subplot(111, projection='3d')
                    # Scatter plot using the coordinates and the brightness as color
                    img = ax.scatter([i[0] for i in coords_inside_cell_1],[i[1] for i in coords_inside_cell_1], points_inside_cell_1, c=points_inside_cell_1, cmap='inferno',s = 10,marker="o")

                    # Adding color bar to represent brightness
                    plt.colorbar(img)
                    fig_3d.savefig(f"Cell/3dplot/{n}.png")
                    fig_3d.savefig(f"3dplot.png")
                    plt.close()
                    #######################################################Min distant point
                    fig_min_point = plt.figure(figsize=[6,6])
                    raw_points = []
                    for i,j in zip(u1,u2):
                        min_distance, min_point = find_minimum_distance_and_point(theta[0],theta[1],theta[2],theta[3],theta[4],i,j)
                        print(min_distance)
                        print(min_point)
                        raw_points.append(min_point)
                        # plt.scatter(min_point[0],min_point[1],s = 10,color = "lime" )
                    projected_points = [[0,raw_points[0][1]]]

                    for i in range(len(raw_points)-1):
                        length = calc_arc_length(theta,raw_points[i][0],raw_points[i+1][0])
                        projected_points.append([projected_points[i][0]+length,raw_points[i+1][1]])
                    temp_y = [points_inside_cell_1[i] for i in range(len(projected_points))]
                    print([round(i[0],2) for i in projected_points],temp_y)
                    plt.scatter([i[0] for i in projected_points],temp_y,c = temp_y ,s = 8,cmap="inferno",marker="x")
                    projected_points_xs.append([i[0] for i in projected_points])
                    projected_points_ys.append(temp_y)
                    plt.xlabel("Distance from the start point (px)")
                    plt.ylabel("Fluo. intensity")
                    plt.ylim(0,255)
                    fig_min_point.savefig(f"Cell/projected_points/{n}.png")
                    plt.close()
                    ##########ピークに沿ったpathの探索アルゴリズム##########
                    data_points = np.array([[i[0],j] for i,j in zip(projected_points,temp_y)])

                    fig_path = plt.figure(figsize=(6, 6))
                    split_num = 140
                    delta_L = (np.max(x) - np.min(x)) / split_num +1
                    x = data_points[:, 0]
                    y = data_points[:, 1]
                    path = []
                    for i in range(split_num):
                        min_x_i = np.min(x) + i * delta_L
                        max_x_i = min_x_i + delta_L
                        
                        indices = (x >= min_x_i) & (x < max_x_i)
                        x_in_range = x[indices]
                        y_in_range = y[indices]

                        if len(y_in_range) > 0 :
                            max_y = np.max(y_in_range)
                            max_y_index = np.argmax(y_in_range)
                            sampled_point = [x_in_range[max_y_index], max_y]
                            path.append(sampled_point)
                    peak_points.append(path)
                    path = np.array(path)
                    print(path)
                    plt.scatter(data_points[:, 0], data_points[:, 1], label='Data Points',s = 20,color = "lime",marker="x")
                    plt.plot(path[:, 0], path[:, 1], color='#FD00FD', label='Path',)
                    plt.scatter(path[:, 0], path[:, 1], color='#FD00FD',s = 10)
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('Path Finder Algorithm')
                    plt.legend()
                    fig_path.savefig(f"Cell/peak_path/{n}.png")
                    fig_path.savefig(f"peak_path.png")
                    plt.close()

                ##########資料作成用(Cell/unified_cells）##########
                
                cell_length_text = f"L={round(cell_length*0.0625,2)}(um)"
                position = (0, 31)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                thickness = 1
                cv2.putText(image_ph, f"{cell_length_text}", position, font, font_scale, font_color, thickness)
                pixel_per_micro_meter = 0.0625
                image_size = image_ph.shape[0]
                scale_bar_length = int(image_size*0.2)
                scale_bar_thickness = int(2*(image_size/100))
                scale_bar_mergins = int(10*(image_size/100))
                scale_bar_color = (255,255,255)
                cv2.rectangle(image_ph,(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins),(image_size-scale_bar_mergins,image_size-scale_bar_mergins-scale_bar_thickness),scale_bar_color,-1)
                position = (image_size-scale_bar_mergins-scale_bar_length, image_size-scale_bar_mergins+scale_bar_thickness+10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                thickness = 1
                cv2.putText(image_ph, f"{round(scale_bar_length*pixel_per_micro_meter,2)} um", position, font, font_scale, font_color, thickness)

                if single_layer_mode:
                    cv2.imwrite(f"Cell/unified_cells/{n}.png",image_ph)
                if not single_layer_mode:
                    if not dual_layer_mode:
                        fluo_out1 = adjust_brightness(fluo_out1,2)
                        cv2.imwrite("fluo1.png",fluo_out1)
                        unify_images_ndarray(image1=image_ph, image2=fluo_out1 ,output_name=f"Cell/unified_cells/{n}")
                if dual_layer_mode:
                    cv2.rectangle(image_fluo2,(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins),(image_size-scale_bar_mergins,image_size-scale_bar_mergins-scale_bar_thickness),scale_bar_color,-1)
                    unify_images_ndarray2(image1=image_ph, image2=fluo_out1, image3=fluo_out2 ,output_name=f"Cell/unified_cells/{n}")
                
    total_rows = int(np.sqrt(n))+ 1
    total_cols = n//total_rows + 1
    num_images = n
    filename = db_name[:-3]
    combine_images_function(total_rows, total_cols, image_size, num_images, out_name,single_layer_mode, dual_layer_mode)

    fig_histo_cumulative_inOne = plt.figure(figsize=[6,6])
    for cumulative_freq in cumulative_frequencys:
        plt.plot(cumulative_freq)
        print(cumulative_freq)
    plt.title('Cumulative Frequency Plot')
    plt.xlabel('Value (0 to 255)')
    plt.ylabel('Cumulative Frequency')
    plt.xlim(-10, 255)
    plt.ylim(0, 1.05)
    plt.grid(True)
    fig_histo_cumulative_inOne.savefig(f"{filename}_cumulative_frequency_one.png")
    plt.close()
    # with open(f"{filename}_cumulative_frequency_one.txt",mode="w") as fpout:
    #     for i in cumulative_frequencys:
    #             fpout.write(f"{','.join([str(float(i)) for i in i])}\n")
    
    with open(f"{out_name}_cell_lengths.txt",mode="w") as fpout:
        for i in cell_lengths:
            fpout.write(f"{i[0]},{i[1]}\n")

    # with open(f"{out_name}_agg_formation_rate.txt",mode="w") as fpout:
    #     fpout.write(f"out_name,num_agg_detected,num_total_cells,agg_form_rate\n")
    #     fpout.write(f"{out_name},{agg_tracker},{n},{agg_tracker/n}\n")

    with open(f"{out_name}_meds_means_vars.txt",mode="w") as fpout:
        for i in range(len(meds)):
            fpout.write(f"{meds[i]},{means[i]},{vars[i]}\n")

    # with open(f"{out_name}_fluo_2_mean_fluo_intensities.txt",mode="w") as fpout:
    #     for i in range(len(mean_fluo_raw_intensities_2)):
    #         fpout.write(f"{mean_fluo_raw_intensities_2[i]}\n")

    # with open(f"{out_name}_max_int_minus_med.txt",mode="w") as fpout:
    #     for i in range(len(meds)):
    #         fpout.write(f"{max_int_minus_med[i]}\n")

    # with open(f"{out_name}_mean_fluo_raw_intensities.txt",mode="w") as fpout:
    #     for i in range(len(meds)):
    #         fpout.write(f"{mean_fluo_raw_intensities[i]}\n")

    # with open(f"{out_name}_energies.txt",mode="w") as fpout:
    #     for i in range(len(energies)):
    #         fpout.write(f"{energies[i][0][0]}\n")
    # with open(f"{out_name}_contrasts.txt",mode="w") as fpout:
    #     for i in range(len(contrasts)):
    #         fpout.write(f"{contrasts[i][0][0]}\n")
    
    # with open(f"{out_name}_dice_similarities.txt",mode="w") as fpout:
    #     for i in range(len(dice_similarities)):
    #         fpout.write(f"{dice_similarities[i]}\n")
    
    # with open(f"{out_name}_homogeneities.txt",mode="w") as fpout:
    #     for i in range(len(homogeneities)):
    #         fpout.write(f"{homogeneities[i][0][0]}\n")
        
    # with open(f"{out_name}_correlations.txt",mode="w") as fpout:
    #     for i in range(len(correlations)):
    #         fpout.write(f"{correlations[i][0][0]}\n")
    
    # with open(f"{out_name}_ASMs.txt",mode="w") as fpout:
    #     for i in range(len(ASMs)):
    #         fpout.write(f"{ASMs[i][0][0]}\n")
    
    # with open(f"{out_name}_smoothnesses.txt",mode="w") as fpout:
    #     for i in range(len(smoothnesses)):
    #         fpout.write(f"{smoothnesses[i]}\n")

    # with open(f"{out_name}_skewnesses.txt",mode="w") as fpout:
    #     for i in range(len(skewnesses)):
    #         fpout.write(f"{skewnesses[i]}\n")

    # with open(f"{out_name}_projected_points_xs.txt",mode="w") as fpout:
    #     for i in range(len(projected_points_xs)):
    #         fpout.write(f"{','.join([str(float(i)) for i in projected_points_xs[i]])}\n")

    # with open(f"{out_name}_projected_points_ys.txt",mode="w") as fpout:
    #     for i in range(len(projected_points_ys)):
    #         fpout.write(f"{','.join([str(float(i)) for i in projected_points_ys[i]])}\n")
    
    # with open(f"{out_name}_kurtosises.txt",mode="w") as fpout:
    #     for i in range(len(kurtosises)):
    #         fpout.write(f"{kurtosises[i]}\n")

    with open(f"{out_name}_peak_points_xs.txt",mode="w") as fpout:
        for i in range(len(peak_points)):
            fpout.write(f"{','.join([str(float(i)) for i in [i[0] for i in peak_points[i]]])}\n")
    
    with open(f"{out_name}_peak_points_ys.txt",mode="w") as fpout:
        for i in range(len(peak_points)):
            fpout.write(f"{','.join([str(float(i)) for i in [i[1] for i in peak_points[i]]])}\n")

    return n
    





        








