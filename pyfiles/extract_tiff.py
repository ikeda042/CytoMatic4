import os 
from PIL import Image

def extract_tiff(tiff_file) -> int:
    folders = [folder for folder in os.listdir("TempData") if os.path.isdir(os.path.join(".", folder))]
    for i in [i for i in ["Fluo1","Fluo2","PH"] if i not in folders]:
        try:
            os.mkdir(f"TempData/{i}")
        except:
            continue
    with Image.open(tiff_file) as tiff:
        num_pages = tiff.n_frames
        img_num = 0
        for i in range(num_pages):
            tiff.seek(i)
            if (i+2)%3 == 0:
                filename = f"TempData/Fluo1/{img_num}.tif"
            elif (i+2)%3 == 1:
                filename = f"TempData//Fluo2/{img_num}.tif"
                img_num += 1
            else:
                filename = f"TempData/PH/{img_num}.tif"
            print(filename)
            tiff.save(filename, format='TIFF')
    return num_pages
