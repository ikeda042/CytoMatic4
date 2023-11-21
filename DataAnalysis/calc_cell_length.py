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
import sympy
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew

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