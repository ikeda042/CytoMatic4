# PhenoPixel4.0

<div align="center">

![Start-up window](app/images_readme/schema.png)

</div>
PhenoPixel4.0 is an OpenCV-based image processing program designed for automating the extraction of cell images from a large number of images (e.g., multiple nd2 files). 



<div align="center">

![Start-up window](app/images_readme/manual_detect_demo.gif)

</div>

It is also capable of detecting the contours of cells manually as shown so that all the phenotypic cells can be equally sampled.

This program is Python-based and utilizes Tkinter for its GUI, making it cross-platform. 

It has been primarily tested on Windows 11 and MacOS Sonoma 14.0.

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



