
import cv2 
import pickle
from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import time
import numpy as np

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

while True:
    engine = create_engine(f"sqlite:///{'app/test_database.db'}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        cells = session.query(Cell).all()
        for cell in cells:
            if cell.manual_label == 1:
                image_ph = cv2.imdecode(
                            np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                        )
                cv2.imwrite("test_ph.png", image_ph)
                image_fluo1 = cv2.imdecode(
                            np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                        )
                cell_contour = [list(i[0]) for i in pickle.loads(cell.contour)]
                
                image_size = image_fluo1.shape
                mask = np.zeros((image_size[0],image_size[1]), dtype=np.uint8)
                cv2.fillPoly(mask, [pickle.loads(cell.contour)], 1)
                output_image = cv2.bitwise_and(image_fluo1, image_fluo1, mask=mask)
                output_image_color = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
                output_image_color[:, :, 0] = 0
                output_image_color[:, :, 2] = 0
                cv2.imwrite("test.png", output_image_color)
            
