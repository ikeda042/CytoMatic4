from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, BLOB, FLOAT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import cv2
import numpy as np
import pickle
import os 
from numpy.linalg import inv, eig
import matplotlib.pyplot as plt

#set the theme dark
plt.style.use('dark_background')
def basis_conversion(contour:list[list[int]],X:np.ndarray,center_x:float,center_y:float,coordinates_incide_cell:list[list[int]]) -> list[list[float]]:
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

class Point:
    def __init__(self, u1: float, G: float):
        self.u1 = u1
        self.G = G
    def __gt__(self, other):
        return self.u1 > other.u1
        

while True:
    engine = create_engine(f'sqlite:///{"app/test_database.db"}', echo=True)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        cells = session.query(Cell).all()
        for cell in cells:
            if cell.manual_label == 1:
                coords_inside_cell_1 = []
                brightness_inside_cell = []
                projected_points: list[Point] = []
                image_fluo = cv2.imdecode(np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                cell_contour = [list(i[0]) for i in pickle.loads(cell.contour)]
                image_size = image_fluo.shape[0]
                for i in range(image_size):
                    for j in range(image_size):
                        if (
                            cv2.pointPolygonTest(
                                pickle.loads(cell.contour), (i, j), False
                            )
                            >= 0
                        ):
                            coords_inside_cell_1.append([i, j])
                            brightness_inside_cell.append(image_fluo[i, j])
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
        
                for u, g in zip(u1, brightness_inside_cell):
                    point = Point(u, g)
                    projected_points.append(point)

                sorted_projected_points = sorted(projected_points, reverse=True)
                fig = plt.figure(figsize=(7,7))
                ax = fig.add_subplot(111)
                ax.scatter(u1, u2, s=10, c="black")
                ax.scatter(u1_c, u2_c, s=10, c="lime")
                ax.scatter(u1_contour, u2_contour, c="lime",s = 20)
                plt.axis("equal")
                ax.set_xlabel("u1")
                ax.set_ylabel("u2")
                ax.set_aspect("equal")
                ax.plot([min_u1, max_u1,], [u2_c,u2_c], c="red", linewidth=2)
                ax.plot([u1_c,u1_c], [min(u2), max(u2)], c="red", linewidth=2)
                ax.scatter(u1_c,max(u2), c="red", s=40)
                ax.scatter(min_u1,u2_c, c="red", s=40)
                ax.scatter(max_u1,u2_c ,c="red", s=40)
                ax.scatter(u1_c,min(u2), c="red", s=40)

                # add second axis
                ax2 = ax.twinx()
                ax2.set_xlabel("u1")
                ax2.set_ylabel("Brightness")
                ax2.set_ylim(0,900)
                ax2.scatter([i.u1 for i in sorted_projected_points],[i.G for i in sorted_projected_points], s=1, c="lime")
                fig.savefig(f"basis_conversion.png",dpi = 300)
                plt.close()
                plt.clf()