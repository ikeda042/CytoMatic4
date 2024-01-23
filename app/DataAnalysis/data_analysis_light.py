from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import numpy as np 
import cv2 
import pickle
import matplotlib.pyplot as plt 
from tqdm import tqdm
from .components import create_dirs
import numpy as np
import seaborn as sns


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

def combine_images_function_light(total_rows, total_cols, image_size, num_images, filename, single_layer_mode, dual_layer_mode):
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
                    image_path = f'Cell/fluo1/{image_index}.png' 
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[i * image_size: (i + 1) * image_size,
                                j * image_size: (j + 1) * image_size] = img
        plt.axis('off')
        cv2.imwrite(f'{filename}_fluo1.png', result_image)


def data_analysis_light(db_name:str = "test.db", image_size:int = 100,out_name:str ="cell",dual_layer_mode:bool = True,single_layer_mode:bool = False):
    ##############################################################
    n = -1
    ##############################################################
    create_dirs(["Cell","Cell/ph","Cell/fluo1","Cell/fluo2","Cell/histo","Cell/histo_cumulative","Cell/replot","Cell/replot_map","Cell/fluo1_incide_cell_only","Cell/fluo2_incide_cell_only","Cell/gradient_magnitudes","Cell/GLCM","Cell/unified_cells","Cell/3dplot","Cell/projected_points","Cell/peak_path"])
    sns.set()
    engine = create_engine(f'sqlite:///{db_name}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        cells = session.query(Cell).all()
    for cell in tqdm(cells):
        if  cell.manual_label != "N/A" and cell.manual_label!= None :
            print("===============================================")
            print(cell.cell_id)
            print("===============================================")
            n+=1
            points_inside_cell_1 = []
            medians = []
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
            image_fluo1 = cv2.imdecode(np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            image_size = image_fluo1.shape
            mask = np.zeros((image_size,image_size), dtype=np.uint8)
            cv2.fillPoly(mask, [pickle.loads(cell.contour)], 1)
            output_image = cv2.bitwise_and(image_fluo1, image_fluo1, mask=mask)
            for i in range(image_size):
                for j in range(image_size):
                    if (
                    cv2.pointPolygonTest(
                        pickle.loads(cell.contour), (i, j), False
                    ) >= 0         
                        ):
                            points_inside_cell_1.append(output_image[j][i])
            fluo_out1 = cv2.imdecode(np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.drawContours(fluo_out1,pickle.loads(cell.contour),-1,(0,255,0),2)
            cv2.imwrite(f"Cell/fluo1/{n}.png",fluo_out1)
            # calc median of points inside cell
            median_fluo1 = np.median(points_inside_cell_1)
            print(f"Median fluo1: {median_fluo1}")
            medians.append(median_fluo1)
            
    total_rows = int(np.sqrt(n))+ 1
    total_cols = n//total_rows + 1
    num_images = n
    combine_images_function_light(total_rows, total_cols, image_size, num_images, out_name,single_layer_mode, dual_layer_mode) 

    with open(f"{out_name}_meds.txt", "w") as f:
        f.write(str(medians))