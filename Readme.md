# PhenoPixel4.0

<div align="center">

![Start-up window](docs_images/Schema_new.png)

</div>
PhenoPixel4.0 is an OpenCV-based image processing program designed for automating the extraction of cell images from a large number of images (e.g., multiple nd2 files). 



<div align="center">

![Start-up window](docs_images/manual_detect_demo.gif)

</div>

It is also capable of detecting the contours of cells manually as shown so that all the phenotypic cells can be equally sampled.

This program is Python-based and utilizes Tkinter for its GUI, making it cross-platform, and has been primarily tested on Windows 11 and MacOS Sonoma 14.0.

# Installation & Setup
1. Install `python 3.8` or higher on your computer.
2. Clone this repository to your computer. (e.g., on visual studio code)
```bash
https://github.com/ikeda042/PhenoPixel4.0.git
```
3. Install the required packages with the following commmand in the root directory of the repository.
```bash
pip install -r app/requirements.txt
```

# Usage
1. Go to the root directory and run `PhenoPixel4.py`
```bash
python PhenoPixel4.py
```
After running the scripts, the landing window automatically pops up. 

![Start-up window](docs_images/landing_window.png)

2. Click "Select File" to choose a file. (file ext must be .nd2/.tif)
   

Input parameters are listed below.

| Parameters | Type | Description |
| :---: | :---: | :--- |
| Parameter 1 | int [0-255] | Lower threshold for Canny algorithm.|
| Parameter 2 | int [0-255] | Higher threshold for Canny algorithm.|
|Image Size | int | Size for square for each cell.|
|Mode| Literal | `all` for general analysis including cell extraction, `Data Analysis` for only data analysis using existing database(.db),  `Data Analysis(all db)` for sequentially read all the databases in the root directly, and `Delete All` to clear unused files.|





