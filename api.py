from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
import uvicorn
from fastapi import FastAPI, File, UploadFile
from typing import cast
import aiofiles
from app.DataAnalysis.load_mat_stackfiles import load_mat
from datetime import datetime
from fastapi import HTTPException
import os
import shutil

app = FastAPI(title="PhenoPixel4.0",docs_url="/phenopixel4.0")

@app.post("/uploadfile/",tags=["ここでスタックファイル(.mat)をアップロード"])
async def create_upload_file(file: UploadFile = File(...)):
    try:
        os.remove("mat_file.mat")
        os.remove("mat_file_overlay.png")
        os.remove("mat_file_heatmap.png")
        shutil.rmtree("Matlab")
    except:
        pass
    print(file.filename,"uploaded+++++++++++++++++++++++++++++++++++++++++++++++")
    print(datetime.now())
    content = await file.read()
    filename = cast(str,file.filename)
    if not filename.endswith('.mat'):
        raise HTTPException(status_code=400, detail="Invalid file extension. Only .mat files are allowed.")
    filename = "mat_file.mat"
    async with aiofiles.open(filename, "wb") as f:
        await f.write(content)
    load_mat(filename)
    # return FileResponse("mat_file_overlay.png")
    return {"filename": file.filename,"status":"file uploaded"}

@app.get("/get-overlay-image",tags=["オーバーレイ画像を取得"])
async def get_overlay():
    print("get_overlay+++++++++++++++++++++++++++++++++++++++++++++++")
    print(datetime.now())
    file_path = "mat_file_overlay.png"  
    return FileResponse(file_path)

@app.get("/get-heatmap",tags=["ヒートマップを取得"])
async def get_heatmap():
    print("get_heatmap+++++++++++++++++++++++++++++++++++++++++++++++")
    print(datetime.now())
    file_path = f"mat_file_heatmap.png"  
    return FileResponse(file_path)

def main():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
if __name__ == "__main__":
    main()