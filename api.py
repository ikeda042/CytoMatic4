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

app = FastAPI(title="PhenoPixel4.0", docs_url="/phenopixel4.0")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


class HeatmapVector:
    def __init__(self, heatmap_vector: np.ndarray, sample_num: int):
        self.heatmap_vector: np.ndarray = heatmap_vector
        self.sample_num: int = sample_num

    def __gt__(self, other):
        self_v = np.sum(self.heatmap_vector)
        other_v = np.sum(other.heatmap_vector)
        return self_v < other_v


def get_heatmap_vector(file: str):
    with open(file, "r") as f:
        ys = [
            [float(x.replace("\n", "")) for x in line.split(",")]
            for line in f.readlines()
        ]
        ys_normalized = []
        for i in ys:
            i = np.array(i)
            i = (i - i.min()) / (i.max() - i.min())
            ys_normalized.append(i.tolist())
    return ys_normalized


def create_heatmap(files: list[str]) -> None:
    vectors = []
    for i, file in enumerate(files):
        vectors += sorted([HeatmapVector(j, i) for j in get_heatmap_vector(file)])

    concatenated_samples = np.column_stack([i.heatmap_vector for i in vectors])

    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[30, 1], height_ratios=[1, 10], hspace=0.05, wspace=0.05
    )
    additional_row = np.array([i.sample_num / 4 for i in vectors])[None, :]
    plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[30, 1], height_ratios=[1, 10], hspace=0.05, wspace=0.05
    )

    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(
        additional_row,
        aspect="auto",
        cmap="inferno",
        extent=[0, concatenated_samples.shape[1], 0, 1],
    )
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = plt.subplot(gs[1, 0])
    im = ax1.imshow(concatenated_samples, aspect="auto", cmap="inferno")
    ax1.set_xlabel(f"Sample Number")
    ax1.set_ylabel("Split index (Relative position)")
    ax2 = plt.subplot(gs[:, 1])
    plt.colorbar(im, cax=ax2)
    ax2.set_ylabel("Normalized fluo. intensity", rotation=270, labelpad=15)
    plt.savefig("heatmap.png")


@app.post("/uploadfile/", tags=["ここでスタックファイル(.mat)をアップロード"])
async def create_upload_file(file: UploadFile = File(...)):
    try:
        os.remove("mat_file.mat")
        os.remove("mat_file_overlay.png")
        os.remove("mat_file_heatmap.png")
        shutil.rmtree("Matlab")
    except:
        pass
    print(file.filename, "uploaded+++++++++++++++++++++++++++++++++++++++++++++++")
    print(datetime.now())
    content = await file.read()
    filename = cast(str, file.filename)
    if not filename.endswith(".mat"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file extension. Only .mat files are allowed.",
        )
    filename = file.filename
    async with aiofiles.open(filename, "wb") as f:
        await f.write(content)
    load_mat(filename)
    # return FileResponse("mat_file_overlay.png")
    return {"filename": file.filename, "status": "file uploaded"}


@app.get("/get-overlay-image", tags=["オーバーレイ画像を取得"])
async def get_overlay():
    print("get_overlay+++++++++++++++++++++++++++++++++++++++++++++++")
    print(datetime.now())
    file_path = "mat_file_overlay.png"
    return FileResponse(file_path)


@app.get("/get-heatmap", tags=["ヒートマップを取得"])
async def get_heatmap():
    print("get_heatmap+++++++++++++++++++++++++++++++++++++++++++++++")
    print(datetime.now())
    file_path = f"mat_file_heatmap.png"
    return FileResponse(file_path)


@app.get("/get-heatmap_all", tags=["全結合ヒートマップを取得"])
async def get_heatmap_all():
    print("get_heatmap_all+++++++++++++++++++++++++++++++++++++++++++++++")
    print(datetime.now())
    file_paths = [
        i
        for i in os.listdir("./")
        if i.split(".")[-1] == "txt" and i.split("_")[-1] == "heatmap.txt"
    ]
    create_heatmap(file_paths)
    return FileResponse("heatmap.png")


@app.delete("/delete-files", tags=["ヒートマップをリセット"])
async def delete_files():
    file_paths = [
        i
        for i in os.listdir("./")
        if i.split(".")[-1] == "txt" and i.split("_")[-1] == "heatmap.txt"
    ]
    for i in file_paths:
        os.remove(i)
    return {"status": "files deleted"}


def main():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
