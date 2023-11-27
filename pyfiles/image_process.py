from .initialize import init
from .unify_images import unify_images_ndarray2, unify_images_ndarray
from .database import  Cell, Base
from .calc_center import get_contour_center
import os
import cv2 
from tqdm import tqdm
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def image_process(input_filename: str = "data.tif",
                    param1: int = 80,
                    param2: int = 255,
                    image_size:int = 100,
                    draw_scale_bar: bool = True,
                    fluo_dual_layer_mode:bool =  True,
                    single_layer_mode:bool = False) -> None:
    engine = create_engine(f'sqlite:///{input_filename.split(".")[0]}.db', echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    num_tif = init(input_filename=input_filename, param1=param1, param2=param2,image_size=image_size,fluo_dual_layer_mode=fluo_dual_layer_mode,single_layer_mode=single_layer_mode)
    for k in tqdm(range(0,num_tif//3)):
        for j in range(len(os.listdir(f'TempData/frames/tiff_{k}/Cells/ph/'))):
            cell_id: str = f"F{k}C{j}"
            img_ph = cv2.imread(f'TempData/frames/tiff_{k}/Cells/ph/{j}.png')
            if not single_layer_mode:
                img_fluo1 = cv2.imread(f'TempData/frames/tiff_{k}/Cells/fluo1/{j}.png')
            
            img_ph_gray = cv2.cvtColor(img_ph,cv2.COLOR_BGR2GRAY)
            if not single_layer_mode:
                img_fluo1_gray = cv2.cvtColor(img_fluo1,cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(img_ph_gray,param1,param2,cv2.THRESH_BINARY)
            img_canny = cv2.Canny(thresh,0,150)
            contours_raw, hierarchy  = cv2.findContours(img_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            #Filter out contours with small area
            contours = list(filter(lambda x: cv2.contourArea(x) >=300 , contours_raw))
            #Check if the center of the contour is not too far from the center of the image
            contours = list(filter(lambda x: abs(cv2.moments(x)['m10']/cv2.moments(x)['m00']-image_size/2) < 10, contours))
            #do the same for y
            contours = list(filter(lambda x: abs(cv2.moments(x)['m01']/cv2.moments(x)['m00']-image_size/2) < 10, contours))
            
            if not single_layer_mode:
                cv2.drawContours(img_fluo1,contours,-1,(0,255,0),1)
                cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/fluo1_contour/{j}.png",img_fluo1)
            cv2.drawContours(img_ph,contours,-1,(0,255,0),1)
            cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/ph_contour/{j}.png",img_ph)
          
         
            if fluo_dual_layer_mode:
                img_fluo2 = cv2.imread(f'TempData/frames/tiff_{k}/Cells/fluo2/{j}.png')
                print(f"empData/frames/tiff_{k}/Cells/fluo2/{j}.png")
                img_fluo2_gray = cv2.cvtColor(img_fluo2,cv2.COLOR_BGR2GRAY)
                cv2.drawContours(img_fluo2,contours,-1,(0,255,0),1)
                cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/fluo2_contour/{j}.png",img_fluo2)
                
            if contours != []:
                if draw_scale_bar:
                    image_ph_copy = img_ph.copy()
                    if not single_layer_mode:
                        image_fluo1_copy = img_fluo1.copy()
                    
                    pixel_per_micro_meter = 0.0625
                    #want to draw a scale bar of 20% of the image width at the bottom right corner. (put some mergins so that the scale bar is not too close to the edge)
                    #scale bar length in pixels
                    scale_bar_length = int(image_size*0.2)
                    scale_bar_size = scale_bar_length*pixel_per_micro_meter
                    #scale bar thickness in pixels
                    scale_bar_thickness = int(2*(image_size/100))
                    #scale bar mergins from the edge of the image
                    scale_bar_mergins = int(10*(image_size/100))
                    #scale bar color
                    scale_bar_color = (255,255,255)
                    #scale bar text color
                    scale_bar_text_color = (255,255,255)
                    #draw scale bar for the both image_ph and image_fluo and the scale bar should be Rectangle
                    #scale bar for image_ph
                    cv2.rectangle(image_ph_copy,(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins),(image_size-scale_bar_mergins,image_size-scale_bar_mergins-scale_bar_thickness),scale_bar_color,-1)
                    # cv2.putText(image_ph_copy,f"{round(scale_bar_size,2)} µm",(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins-2*scale_bar_thickness),cv2.FONT_HERSHEY_SIMPLEX,0.2,scale_bar_text_color,1,cv2.LINE_AA)
                    #scale bar for image_fluo
                    if not single_layer_mode:
                        cv2.rectangle(image_fluo1_copy,(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins),(image_size-scale_bar_mergins,image_size-scale_bar_mergins-scale_bar_thickness),scale_bar_color,-1)
                    # cv2.putText(image_fluo_copy,f"{round(scale_bar_size,2)} µm",(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins-2*scale_bar_thickness),cv2.FONT_HERSHEY_SIMPLEX,0.2,scale_bar_text_color,1,cv2.LINE_AA)
                    if fluo_dual_layer_mode:
                        image_fluo2_copy = img_fluo2.copy()
                        cv2.rectangle(image_fluo2_copy,(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins),(image_size-scale_bar_mergins,image_size-scale_bar_mergins-scale_bar_thickness),scale_bar_color,-1)
                        unify_images_ndarray2(image1=image_ph_copy, image2=image_fluo1_copy, image3=image_fluo2_copy ,output_name=f"TempData/frames/tiff_{k}/Cells/unified_images/{j}")
                        unify_images_ndarray2(image1=image_ph_copy, image2=image_fluo1_copy, image3=image_fluo2_copy ,output_name=f"TempData/app_data/{cell_id}")
                    elif single_layer_mode:
                        cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/unified_images/{j}.png",image_ph_copy)
                        cv2.imwrite(f"TempData/app_data/{cell_id}.png",image_ph_copy)
                    else:
                        unify_images_ndarray(image1=image_ph_copy, image2=image_fluo1_copy ,output_name=f"TempData/frames/tiff_{k}/Cells/unified_images/{j}")
                        unify_images_ndarray(image1=image_ph_copy, image2=image_fluo1_copy ,output_name=f"TempData/app_data/{cell_id}")

                with Session() as session:
                    perimeter = cv2.arcLength(contours[0], closed=True)
                    area = cv2.contourArea(contour=contours[0])
                    image_ph_data = cv2.imencode('.png', img_ph_gray)[1].tobytes()
                    if not single_layer_mode:
                        image_fluo1_data = cv2.imencode('.png', img_fluo1_gray)[1].tobytes()
                    if fluo_dual_layer_mode:
                        image_fluo2_data = cv2.imencode('.png', img_fluo2_gray)[1].tobytes()
                    contour = pickle.dumps(contours[0])
                    center_x, center_y = get_contour_center(contours[0])
                    print(center_x,center_y)
                    if fluo_dual_layer_mode:
                        cell = Cell(cell_id=cell_id, label_experiment="", perimeter=perimeter, area=area, img_ph=image_ph_data, img_fluo1=image_fluo1_data, img_fluo2 =image_fluo2_data ,contour=contour, center_x=center_x, center_y=center_y)
                    elif single_layer_mode:
                        cell = Cell(cell_id=cell_id, label_experiment="", perimeter=perimeter, area=area, img_ph=image_ph_data, contour=contour, center_x=center_x, center_y=center_y)
                    else:
                        cell = Cell(cell_id=cell_id, label_experiment="", perimeter=perimeter, area=area, img_ph=image_ph_data, img_fluo1=image_fluo1_data, contour=contour, center_x=center_x, center_y=center_y)
                    if session.query(Cell).filter_by(cell_id = cell_id).first() is None:
                        session.add(cell)
                        session.commit()



