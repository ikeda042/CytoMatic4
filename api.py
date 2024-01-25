from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
import uvicorn
from fastapi import FastAPI, File, UploadFile
from typing import cast
import aiofiles
from app.DataAnalysis.load_mat_stackfiles import load_mat

app = FastAPI(title="PhenoPixel4.0")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = cast(str,file.filename)
    filename = "mat_file.mat"
    async with aiofiles.open(filename, "wb") as f:
        await f.write(content)
    load_mat(filename)
    file_path = f"{filename.replace('.mat','')}_heatmap.png"  
    return {"filename": filename}

@app.get("/get-overlay-image")
async def get_overlay():
    file_path = "mat_file_overlay.png"  
    return FileResponse(file_path)

@app.get("/get-heatmap")
async def get_heatmap():
    file_path = f"mat_file_heatmap.png"  
    return FileResponse(file_path)

def main():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
if __name__ == "__main__":
    main()