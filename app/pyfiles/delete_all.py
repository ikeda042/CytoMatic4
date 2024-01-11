import os
import shutil

def delete_all():
    dirs = ["app_data","Fluo","frames","manual_detection_data","manual_detection_data_raw","PH","ph_contours","Fluo1","Fluo2"]
    if "TempData" in os.listdir():
        for i in [f"TempData/{i}" for i in dirs if i in os.listdir("TempData")] + ["ph_contours"]:
            try:
                shutil.rmtree(f"{i}")
            except Exception as e:
                print(e)
    try :
        shutil.rmtree("Cell")
    except Exception as e:
        print(e)
    try :
        shutil.rmtree("nd2totiff")
    except Exception as e:
        print(e)

    try:
        os.remove("cell.db")
    except Exception as e:
        print(e)
    try:
        os.remove("image_labels.db")
    except Exception as e:
        print(e)
