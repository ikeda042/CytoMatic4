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


def nucleoid_analysis(db_name:str) -> None:
    create_dirs(["Cell","Cell/ph","Cell/fluo1","Cell/fluo2","Cell/histo","Cell/histo_cumulative","Cell/replot","Cell/replot_map","Cell/fluo1_incide_cell_only","Cell/fluo2_incide_cell_only","Cell/gradient_magnitudes","Cell/GLCM","Cell/unified_cells","Cell/3dplot","Cell/projected_points","Cell/peak_path"])
    engine = create_engine(f'sqlite:///{db_name}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        cells = session.query(Cell).all()
        image_ph = cv2.imdecode(np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_COLOR)
        image_ph_copy = image_ph.copy()
        image_size = image_ph.shape[0]
        image_fluo1 = cv2.imdecode(np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        fluo_out1 = cv2.imdecode(np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.drawContours(fluo_out1,pickle.loads(cell.contour),-1,(0,0,255),1)
        cv2.imwrite(f"Cell/fluo1/{n}.png",fluo_out1)
        output_image =  np.zeros((image_size,image_size),dtype=np.uint8)
        
        coords_inside_cell_1,  points_inside_cell_1 = [], []
        coords_inside_cell_2,  points_inside_cell_2 = [], []

        for cell in tqdm(cells):
            if  cell.manual_label != "N/A" and cell.manual_label!= None:
                for i in range(image_size):
                    for j in range(image_size):
                        if cv2.pointPolygonTest(pickle.loads(cell.contour), (i,j), False)>=0:
                            coords_inside_cell_1.append([i,j])
                            points_inside_cell_1.append(output_image[j][i])
        
            


