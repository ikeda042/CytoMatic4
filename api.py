from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
import uvicorn
from fastapi import FastAPI, File, UploadFile
from typing import cast
import aiofiles
from app.DataAnalysis.load_mat_stackfiles import load_mat
from datetime import datetime

app = FastAPI(title="PhenoPixel4.0",docs_url="phenopixel4.0")

@app.post("/uploadfile/",tags=["ここでスタックファイル(.mat)をアップロード"])
async def create_upload_file(file: UploadFile = File(...)):
    print(file.filename,"uploaded+++++++++++++++++++++++++++++++++++++++++++++++")
    print(datetime.now())
    content = await file.read()
    filename = cast(str,file.filename)
    filename = "mat_file.mat"
    async with aiofiles.open(filename, "wb") as f:
        await f.write(content)
    load_mat(filename)
    file_path = f"{filename.replace('.mat','')}_heatmap.png"  
    return {"filename": filename}

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