import os 
import numpy as np
import cv2 
from numpy.linalg import eig
import matplotlib.pyplot as plt
import scipy 

def create_dirs(dirs:list[str]) -> None:
    for i in dirs:
        if not os.path.exists(i):
            os.makedirs(i)
            print("Directory created: " + i)
        else:
            print("Directory " + i + " already exists")

def calc_gradient(image_array:np.ndarray) -> None:
    #勾配計算
    # Sobelフィルタを適用してX方向の勾配を計算
    sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)

    # Sobelフィルタを適用してY方向の勾配を計算
    sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)

    # 勾配の合成（勾配強度と角度を計算）
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 勾配の強度を正規化
    # gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 勾配強度画像を保存
    cv2.imwrite(f'Cell/gradient_magnitudes/gradient_magnitude{n}.png', gradient_magnitude)

def basis_conversion(contour:list[list[int]],X:np.ndarray,center_x:float,center_y:float,coordinates_incide_cell:list[list[int]]) -> list[list[float]]:
    print(X)
    Sigma = np.cov(X)
    eigenvalues, eigenvectors = eig(Sigma)
    if eigenvalues[1] < eigenvalues[0]:
        m = eigenvectors[1][1]/eigenvectors[1][0]
        Q = np.array([eigenvectors[1],eigenvectors[0]])
        U = [Q.transpose()@np.array([i,j]) for i,j in coordinates_incide_cell]
        U = [[j,i] for i,j in U]
        contour_U = [Q.transpose()@np.array([j,i]) for i,j in contour]
        contour_U = [[j,i] for i,j in contour_U]
        color = "red"
        center = [center_x, center_y]
        u1_c, u2_c = center@Q
    else:
        m = eigenvectors[0][1]/eigenvectors[0][0]
        Q = np.array([eigenvectors[0],eigenvectors[1]])
        U = [Q.transpose()@np.array([j,i]).transpose() for i,j in coordinates_incide_cell]
        contour_U = [Q.transpose()@np.array([i,j]) for i,j in contour]
        color = "blue"
        center = [center_x,
                  center_y]
        u2_c, u1_c = center@Q
    
    u1 = [i[1] for i in U]
    u2 = [i[0] for i in U]
    u1_contour = [i[1] for i in contour_U]
    u2_contour = [i[0] for i in contour_U]
    min_u1, max_u1 = min(u1), max(u1)
    return u1,u2,u1_contour,u2_contour,min_u1,max_u1,u1_c,u2_c, U, contour_U
    

def calc_arc_length(theta:list[float],u_1_1:float,u_1_2:float) -> float:
    fx = lambda t: np.sqrt((4*theta[0]*t**3 + 3*theta[1]*t**2 + 2*theta[2]*t + theta[3])**2 + 1)
    cell_length, _ = scipy.integrate.quad(fx, u_1_1, u_1_2, epsabs=1e-01)
    return cell_length

