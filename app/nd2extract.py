import nd2reader
import numpy as np
from PIL import Image
import os


def extract_nd2(file_name: str):
    """
    nd2ファイルをMultipageTIFFに変換する。
    """
    try:
        os.mkdir("nd2totiff")
    except:
        pass
    with nd2reader.ND2Reader(file_name) as images:
        images.bundle_axes = "cyx"  # z: depth, c: channel, y: height, x: width
        images.iter_axes = (
            "v"  # iterate over 'v' axis, which could be time, series, etc.
        )
        print(f"Total images: {len(images)}")
        print(f"Channels: {images.sizes['c']}")
        channels = images.metadata["channels"]
        for n, img in enumerate(images):
            for channel in range(images.sizes["c"]):
                array = np.array(img[channel])
                array = array.astype(np.float32)  # Convert to float
                array -= array.min()  # Normalize to 0
                array /= array.max()  # Normalize to 1
                array *= 255  # Scale to 0-255
                array = array.astype(np.uint8)  # Convert to 8-bit integer
                image = Image.fromarray(array)
                image.save(f"nd2totiff/image_{n}_channel_{channel}.tif")
        all_images = []
        for i in range(n):
            for j in range(len(channels)):
                all_images.append(Image.open(f"nd2totiff/image_{i}_channel_{j}.tif"))
        all_images[0].save(
            f"{file_name.split('/')[-1].split('.')[0]}.tif",
            save_all=True,
            append_images=all_images[1:],
        )
