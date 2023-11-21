from pyfiles.image_process import image_process
from pyfiles.delete_all import delete_all
from pyfiles.app import app
from DataAnalysis.data_analysis import data_analysis
import sqlite3
from pyfiles.database import Base, Cell
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker
from typing import Literal
from nd2extract import extract_nd2

#Parameters to specify
#####################################################
file_name = "data.tif"
param1 = 100
param2 = 255
img_size = 200
mode: Literal["all","data_analysis","delete_all"] = "all"
dual_layer_mode = False
single_layer_mode = True
nd2_extract = False
nd2_filename = "None"
#####################################################

if __name__ == "__main__":
    if nd2_extract:
        extract_nd2(nd2_filename)
    if mode == "all":
        image_process(input_filename=file_name, param1=param1, param2=param2,image_size=img_size,fluo_dual_layer_mode=dual_layer_mode,single_layer_mode=single_layer_mode)
        app()
        conn = sqlite3.connect('image_labels.db')
        cursor = conn.cursor()
        table_name = 'labels'
        columns = ['id', 'image_id', 'label'] 
        query = f"SELECT {', '.join(columns)} FROM {table_name}"
        cursor.execute(query)
        cells = cursor.fetchall()
        for cell in cells:
            print(cell)
        engine = create_engine(f'sqlite:///{file_name.split(".")[0]}.db', echo=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            for cell in cells:
                stmt = update(Cell).where(Cell.cell_id == cell[1]).values(manual_label = cell[2])
                session.execute(stmt)
                session.commit()
    elif mode == "data_analysis":
        data_analysis(db_name=f"{file_name.split('.')[0]}.db", image_size=img_size,out_name = file_name.split(".")[0],dual_layer_mode=dual_layer_mode)
    # elif mode == "delete_all":
    #     delete_all(input_filename=file_name)
    # else:
    #     raise ValueError("mode must be all, data_analysis or delete_all")

