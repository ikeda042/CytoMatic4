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
from tqdm import tqdm
from combine_images import combine_images_function
import scipy.integrate
from components import create_dirs

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
    img_fluo2 = Column(BLOB)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)

def calculate_cell_length(db_name:str):
    try:
        os.mkdir("Cell")
    except:
        pass
    try:
        os.mkdir("Cell/ph_1")
    except:
        pass
    try:
        os.mkdir("Cell/ph_2")
    except:
        pass
    try:
        os.mkdir("Cell/replot_1")
    except:
        pass
    try:
        os.mkdir("Cell/replot_2")
    except:
        pass
    cell_lengths_1, cell_lengths_2 = [], []

    engine = create_engine(f'sqlite:///{db_name}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        cells = session.query(Cell).all()
        n_1, n_2 = -1, -1
        image_for_size = cv2.imdecode(np.frombuffer(cells[0].img_ph, dtype=np.uint8), cv2.IMREAD_COLOR)
        image_size = image_for_size.shape[0]
        #ラベル1について解析する。
        for cell in tqdm(cells):
            if cell.manual_label == 1:
                n_1+= 1
                image_ph = cv2.imdecode(np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_COLOR)
                image_ph_copy = image_ph.copy()
                cv2.drawContours(image_ph_copy,pickle.loads(cell.contour),-1,(0,255,0),1)
                position = (0, 15)  
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5 
                font_color = (255, 255, 255)  
                thickness = 1 
                cv2.putText(image_ph, f"{cell.cell_id}", position, font, font_scale, font_color, thickness)
                cv2.imwrite(f"Cell/ph_1/{n_1}.png",image_ph_copy)
                contour = [[j,i] for i,j in [i[0] for i in pickle.loads(cell.contour)]]
                coords_inside_cell_1 = []
                for i in range(image_size):
                    for j in range(image_size):
                        if cv2.pointPolygonTest(pickle.loads(cell.contour), (i,j), False)>=0:
                                    coords_inside_cell_1.append([i,j])
                X = np.array([[i[1] for i in coords_inside_cell_1],[i[0] for i in coords_inside_cell_1]])
                Sigma = np.cov(X)
                eigenvalues, eigenvectors = eig(Sigma)
                if eigenvalues[1] < eigenvalues[0]:
                    m = eigenvectors[1][1]/eigenvectors[1][0]
                    Q = np.array([eigenvectors[1],eigenvectors[0]])
                    U = [Q.transpose()@np.array([i,j]) for i,j in coords_inside_cell_1]
                    U = [[j,i] for i,j in U]
                    contour_U = [Q.transpose()@np.array([j,i]) for i,j in contour]
                    contour_U = [[j,i] for i,j in contour_U]
                    color = "red"
                    center = [cell.center_x,cell.center_y]
                    u1_c, u2_c = center@Q
                else:
                    m = eigenvectors[0][1]/eigenvectors[0][0]
                    Q = np.array([eigenvectors[0],eigenvectors[1]])
                    U = [Q.transpose()@np.array([j,i]).transpose() for i,j in coords_inside_cell_1]
                    contour_U = [Q.transpose()@np.array([i,j]) for i,j in contour]
                    color = "blue"
                    center = [cell.center_x,cell.center_y]
                    u2_c, u1_c = center@Q

                u1 = [i[1] for i in U]
                u2 = [i[0] for i in U]
                u1_contour = [i[1] for i in contour_U]
                u2_contour = [i[0] for i in contour_U]
                min_u1, max_u1 = min(u1), max(u1)

                fig = plt.figure(figsize=[6,6])
                cmap = plt.get_cmap('inferno')
                x = np.linspace(0,100,1000)

                W = np.array([[i**4,i**3,i**2,i,1] for i in [i[1] for i in U]])
                f = np.array([i[0] for i in U])

                theta = inv(W.transpose()@W)@W.transpose()@f
                x = np.linspace(min_u1,max_u1,1000)
                y = [theta[0]*i**4+theta[1]*i**3 + theta[2]*i**2+theta[3]*i + theta[4] for i in x]

                #弧長積分
                fx = lambda t: np.sqrt((4*theta[0]*t**3 + 3*theta[1]*t**2 + 2*theta[2]*t + theta[3])**2 + 1)
                cell_length, _ = scipy.integrate.quad(fx, min_u1, max_u1,epsabs=1e-2)
                print(cell_length,"Scipy")
                cell_lengths_1.append([cell.cell_id,cell_length])
                plt.plot(x,y,color = "blue",linewidth=1)
                plt.scatter(min_u1,theta[0]*min_u1**4+theta[1]*min_u1**3 + theta[2]*min_u1**2+theta[3]*min_u1 + theta[4],s = 100,color = "red",zorder = 100,marker = "x")
                plt.scatter(max_u1,theta[0]*max_u1**4+theta[1]*max_u1**3 + theta[2]*max_u1**2+theta[3]*max_u1 + theta[4],s = 100,color = "red",zorder = 100,marker = "x")
                plt.xlim(min_u1-40,max_u1+40)
                plt.ylim(u2_c-40,u2_c+40)
                plt.scatter(u1_contour,u2_contour,s = 10,color = "lime" )
                plt.grid()
                plt.xlabel("u1")
                plt.ylabel("u2")
                plt.axis("equal")    
                plt.text(0.5, 0.5, f"cell length = {cell_length}", size=10, ha="center", va="center",bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
                fig.savefig(f"Cell/replot_1/{n_1}.png")
                plt.close()
            elif cell.manual_label == 2:
                n_2+= 1
                image_ph = cv2.imdecode(np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_COLOR)
                image_ph_copy = image_ph.copy()
                cv2.drawContours(image_ph_copy,pickle.loads(cell.contour),-1,(0,255,0),1)
                position = (0, 15)  
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5 
                font_color = (255, 255, 255)  
                thickness = 1 
                cv2.putText(image_ph, f"{cell.cell_id}", position, font, font_scale, font_color, thickness)
                cv2.imwrite(f"Cell/ph_2/{n_2}.png",image_ph_copy)
                contour = [[j,i] for i,j in [i[0] for i in pickle.loads(cell.contour)]]
                coords_inside_cell_1 = []
                for i in range(image_size):
                    for j in range(image_size):
                        if cv2.pointPolygonTest(pickle.loads(cell.contour), (i,j), False)>=0:
                                    coords_inside_cell_1.append([i,j])
                X = np.array([[i[1] for i in coords_inside_cell_1],[i[0] for i in coords_inside_cell_1]])
                Sigma = np.cov(X)
                eigenvalues, eigenvectors = eig(Sigma)
                if eigenvalues[1] < eigenvalues[0]:
                    m = eigenvectors[1][1]/eigenvectors[1][0]
                    Q = np.array([eigenvectors[1],eigenvectors[0]])
                    U = [Q.transpose()@np.array([i,j]) for i,j in coords_inside_cell_1]
                    U = [[j,i] for i,j in U]
                    contour_U = [Q.transpose()@np.array([j,i]) for i,j in contour]
                    contour_U = [[j,i] for i,j in contour_U]
                    color = "red"
                    center = [cell.center_x,cell.center_y]
                    u1_c, u2_c = center@Q
                else:
                    m = eigenvectors[0][1]/eigenvectors[0][0]
                    Q = np.array([eigenvectors[0],eigenvectors[1]])
                    U = [Q.transpose()@np.array([j,i]).transpose() for i,j in coords_inside_cell_1]
                    contour_U = [Q.transpose()@np.array([i,j]) for i,j in contour]
                    color = "blue"
                    center = [cell.center_x,cell.center_y]
                    u2_c, u1_c = center@Q

                u1 = [i[1] for i in U]
                u2 = [i[0] for i in U]
                u1_contour = [i[1] for i in contour_U]
                u2_contour = [i[0] for i in contour_U]
                min_u1, max_u1 = min(u1), max(u1)

                fig = plt.figure(figsize=[6,6])
                cmap = plt.get_cmap('inferno')
                x = np.linspace(0,100,1000)

                W = np.array([[i**4,i**3,i**2,i,1] for i in [i[1] for i in U]])
                f = np.array([i[0] for i in U])

                theta = inv(W.transpose()@W)@W.transpose()@f
                x = np.linspace(min_u1,max_u1,1000)
                y = [theta[0]*i**4+theta[1]*i**3 + theta[2]*i**2+theta[3]*i + theta[4] for i in x]

                #弧長積分
                fx = lambda t: np.sqrt((4*theta[0]*t**3 + 3*theta[1]*t**2 + 2*theta[2]*t + theta[3])**2 + 1)
                cell_length, _ = scipy.integrate.quad(fx, min_u1, max_u1)
                cell_lengths_2.append([cell.cell_id,cell_length])
                plt.plot(x,y,color = "blue",linewidth=1)
                plt.scatter(min_u1,theta[0]*min_u1**4+theta[1]*min_u1**3 + theta[2]*min_u1**2+theta[3]*min_u1 + theta[4],s = 100,color = "red",zorder = 100,marker = "x")
                plt.scatter(max_u1,theta[0]*max_u1**4+theta[1]*max_u1**3 + theta[2]*max_u1**2+theta[3]*max_u1 + theta[4],s = 100,color = "red",zorder = 100,marker = "x")
                plt.xlim(min_u1-10,max_u1+10)
                plt.ylim(u2_c-40,u2_c+40)
                plt.scatter(u1_contour,u2_contour,s = 10,color = "lime")
                plt.grid()
                fig.savefig(f"Cell/replot_2/{n_2}.png")
                plt.close()
                
        ##n1の画像をまとめる
        total_rows = int(np.sqrt(n_1))
        total_cols = total_rows + 1
        num_images = n_1
        filename = db_name.split(".")[0] + "_ph_1"
        
        result_image = np.zeros((total_rows * image_size, total_cols * image_size, 3), dtype=np.uint8)
        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j  # 画像のインデックス
                if image_index <num_images:
                    image_path = f'Cell/replot_1/{image_index}.png'  # 画像のパスを適切に設定
                    print(image_path)
                    # 画像を読み込んでリサイズ
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    # まとめる画像に配置
                    result_image[i * image_size: (i + 1) * image_size,
                                j * image_size: (j + 1) * image_size] = img

        plt.axis('off')
        # まとめた画像を保存
        cv2.imwrite(f'{filename}_replot_1.png', result_image)
        plt.close()

        ##n2の画像をまとめる
        filename = db_name.split(".")[0] + "_ph_2"
        total_rows = int(np.sqrt(n_2))
        total_cols = total_rows + 1
        num_images = n_2
        result_image = np.zeros((total_rows * image_size, total_cols * image_size, 3), dtype=np.uint8)
        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j  # 画像のインデックス
                if image_index <num_images:
                    image_path = f'Cell/replot_2/{image_index}.png'  # 画像のパスを適切に設定
                    print(image_path)
                    # 画像を読み込んでリサイズ
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    # まとめる画像に配置
                    result_image[i * image_size: (i + 1) * image_size,
                                j * image_size: (j + 1) * image_size] = img

        plt.axis('off')
        # まとめた画像を保存
        cv2.imwrite(f'{filename}_replot_2.png', result_image)
        plt.close()
        
        with open(f"{ db_name.split('.')[0]}_cell_lengths_1.txt","w") as f:
            for i in cell_lengths_1:
                f.write(f"{i[0]},{i[1]}\n")
        with open(f"{ db_name.split('.')[0]}_cell_lengths_2.txt","w") as f:
            for i in cell_lengths_2:
                f.write(f"{i[0]},{i[1]}\n")




    
        
def list_files(directory):
    return os.listdir(directory)

# 使用例
directory_path = './'
files_and_dirs = [i for i in list_files(directory_path) if i.split(".")[-1] == "db"]
print(files_and_dirs)
for i in files_and_dirs:
    calculate_cell_length(i)

