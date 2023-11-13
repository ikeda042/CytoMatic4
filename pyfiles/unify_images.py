import cv2
import numpy as np

def unify_images_ndarray2(image1, image2, image3, output_name):
    combined_width = image1.shape[1] + image2.shape[1] + image3.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0], image3.shape[0])
    
    canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Image 1
    canvas[:image1.shape[0], :image1.shape[1]] = image1

    # Image 2
    offset_x_image2 = image1.shape[1]
    canvas[:image2.shape[0], offset_x_image2:offset_x_image2+image2.shape[1]] = image2

    # Image 3
    offset_x_image3 = offset_x_image2 + image2.shape[1]
    canvas[:image3.shape[0], offset_x_image3:offset_x_image3+image3.shape[1]] = image3

    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def unify_images_ndarray(image1, image2, output_name):
    combined_width = image1.shape[1] + image2.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0])
    
    canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    canvas[:image1.shape[0], :image1.shape[1], :] = image1
    canvas[:image2.shape[0], image1.shape[1]:, :] = image2
    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
