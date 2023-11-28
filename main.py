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

def main(
        file_name:str,
        param1:int,
        param2:int,
        img_size:int,
        mode: Literal["all","data_analysis","delete_all"] = "all",
        layer_mode: Literal["dual","single","normal"] = "dual",
        ):
    delete_all()
    if layer_mode == "dual":
        dual_layer_mode = True
        single_layer_mode = False
    elif layer_mode == "single":
        dual_layer_mode = False
        single_layer_mode = True
    else:
        dual_layer_mode = False
        single_layer_mode = False
    
    if mode == "all":
        file_name = f'{"".join(file_name.split(".")[0:-1])}.{file_name.split(".")[-1]}'
        if file_name.split(".")[-1] == "nd2":
            extract_nd2(file_name)
            file_name = file_name.split('/')[-1].split(".")[0] + ".tif"
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
        data_analysis(db_name=f"{file_name.split('.')[0]}.db", image_size=img_size,out_name = file_name.split(".")[0],single_layer_mode=single_layer_mode, dual_layer_mode=dual_layer_mode)
    elif mode == "data_analysis":
        data_analysis(db_name=f"{file_name.split('.')[0]}.db", image_size=img_size,out_name = file_name.split(".")[0],single_layer_mode=single_layer_mode,dual_layer_mode=dual_layer_mode)
    # elif mode == "delete_all":
    #     delete_all(input_filename=file_name)
    elif mode == "delete_all":
        delete_all()
    else:
        data_analysis(db_name=f"{file_name.split('.')[0]}.db", image_size=img_size,out_name = file_name.split(".")[0],single_layer_mode=single_layer_mode, dual_layer_mode=dual_layer_mode)

#Parameters to specify
#####################################################
file_name = "sk328tri30min.db"
param1 = 130
param2 = 255
img_size = 500
mode: Literal["all","data_analysis","delete_all"] = "data_analysis"
layer_mode: Literal["dual","single","normal"] = "normal"
#####################################################
# import os
# for file_name  in [i for i in os.listdir() if i.split(".")[-1] == "db"]:
#     main(file_name, param1, param2, img_size, mode, layer_mode)
