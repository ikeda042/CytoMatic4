from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from fastapi import FastAPI, File, UploadFile
from typing import List
import aiofiles

app = FastAPI(title="PhenoPixel4.0", version="0.0.0")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    async with aiofiles.open(content, "wb") as f:
        await f.write(content)
    return {"filename": file.filename, "content": content.decode()}

@app.get("/get-overlay-image")
async def get_overlay():
    file_path = "Ph_com_mesh_signal_overlay.png"  
    return FileResponse(file_path)

@app.get("/get-heatmap")
async def get_heatmap():
    file_path = "Ph_com_mesh_signal_heatmap.png"  
    return FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=True)
