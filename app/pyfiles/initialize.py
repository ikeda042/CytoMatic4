from .extract_tiff import extract_tiff
from .crop_contours import crop_contours
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def init(input_filename: str,
         param1: int = 140,
         param2: int = 255,
         image_size:int = 100,
         fluo_dual_layer_mode:bool = True,
         single_layer_mode:bool = False) -> int:
    
    if fluo_dual_layer_mode:
        set_num = 3
        init_folders = ["Fluo1", "Fluo2", "PH","frames","app_data"]
    elif single_layer_mode:
        set_num = 1
        init_folders = ["PH","frames","app_data"]
    else:
        set_num = 2
        init_folders = ["Fluo1", "PH","frames","app_data"]

    try:
        os.mkdir("TempData")
    except:
        pass
    
    init_folders = [f"TempData/{d}" for d in init_folders]
    folders = [folder for folder in os.listdir("TempData") if os.path.isdir(os.path.join(".", folder))]
    for i in [i for i in init_folders if i not in folders]:
        try:
            os.mkdir(f"{i}")
        except:
            continue
    
    #画像の枚数を取得
    num_tif = extract_tiff(input_filename,fluo_dual_layer=fluo_dual_layer_mode,singe_layer_mode=single_layer_mode)
    #フォルダの作成
    for i in tqdm(range(num_tif//set_num)):
        try:
            os.mkdir(f"TempData/frames/tiff_{i}")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"TempData/frames/tiff_{i}/Cells")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"TempData/frames/tiff_{i}/Cells/ph")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"TempData/frames/tiff_{i}/Cells/ph_raw")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo1")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo1_adjusted")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"TempData/frames/tiff_{i}/Cells/ph_contour")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo1_contour")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"TempData/frames/tiff_{i}/Cells/unified_images")
        except Exception as e:
            print(e)
        try:
            os.mkdir(f"ph_contours")
        except Exception as e:
            print(e)

        if fluo_dual_layer_mode:
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo2")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo2_adjusted")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo2_contour")
            except Exception as e:
                print(e)

    
    for k in tqdm(range(num_tif//set_num)):
        print(f"TempData/PH/{k}.tif")
        image_ph = cv2.imread(f"TempData/PH/{k}.tif")
        image_fluo_1 = cv2.imread(f"TempData/Fluo1/{k}.tif")
        if fluo_dual_layer_mode:
            image_fluo_2 = cv2.imread(f"TempData/Fluo2/{k}.tif")
        img_gray = cv2.cvtColor(image_ph, cv2.COLOR_BGR2GRAY)

        #２値化を行う
        ret, thresh = cv2.threshold(img_gray, param1, param2, cv2.THRESH_BINARY)
        img_canny = cv2.Canny(thresh, 0, 130)

        contours, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #細胞の面積で絞り込み
        contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours))
        #中心座標が画像の中心から離れているものを除外
        contours = list(filter(lambda x: cv2.moments(x)["m10"] / cv2.moments(x)["m00"] > 400 and cv2.moments(x)["m10"] / cv2.moments(x)["m00"] < 1700, contours))
        # do the same for y
        contours = list(filter(lambda x: cv2.moments(x)["m01"] / cv2.moments(x)["m00"] > 400 and cv2.moments(x)["m01"] / cv2.moments(x)["m00"] < 1700, contours))
        
        output_size = (image_size, image_size)

        if not single_layer_mode:
            cropped_images_fluo_1 = crop_contours(image_fluo_1, contours, output_size)
        if fluo_dual_layer_mode:
            cropped_images_fluo_2 = crop_contours(image_fluo_2, contours, output_size)
        cropped_images_ph = crop_contours(image_ph, contours, output_size)

        image_ph_copy = image_ph.copy()
        cv2.drawContours(image_ph_copy,contours,-1,(0,255,0),3)
        cv2.imwrite(f"ph_contours/{k}.png",image_ph_copy)
        n = 0
        if fluo_dual_layer_mode:
            for j, ph, fluo1, fluo2 in zip([i for i in range(len(cropped_images_ph))], cropped_images_ph, cropped_images_fluo_1, cropped_images_fluo_2):
                if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/ph/{n}.png', ph)
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/fluo1/{n}.png', fluo1)
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/fluo2/{n}.png', fluo2)
                    brightness_factor_fluo1 = 255/ np.max(fluo1)
                    image_fluo1_brightened =  cv2.convertScaleAbs(fluo1, alpha=brightness_factor_fluo1 , beta=0)
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/fluo_adjusted/{n}.png', image_fluo1_brightened)
                    brightness_factor_fluo2 = 255/ np.max(fluo2)
                    image_fluo2_brightened =  cv2.convertScaleAbs(fluo2, alpha=brightness_factor_fluo2 , beta=0)
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/fluo_adjusted/{n}.png', image_fluo2_brightened)
                    n += 1
        elif single_layer_mode:
            for j, ph in zip([i for i in range(len(cropped_images_ph))], cropped_images_ph):
                if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/ph/{n}.png', ph)
                    n += 1
        else:
            for j, ph, fluo1 in zip([i for i in range(len(cropped_images_ph))], cropped_images_ph, cropped_images_fluo_1):
                  if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/ph/{n}.png', ph)
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/fluo1/{n}.png', fluo1)
                    brightness_factor_fluo1 = 255/ np.max(fluo1)
                    image_fluo1_brightened =  cv2.convertScaleAbs(fluo1, alpha=brightness_factor_fluo1 , beta=0)
                    cv2.imwrite(f'TempData/frames/tiff_{k}/Cells/fluo_adjusted/{n}.png', image_fluo1_brightened)
                    n += 1
        
    return num_tif