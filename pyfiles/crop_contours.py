import cv2 
import numpy as np
def get_contour_center(contour):
    # 輪郭のモーメントを計算して重心を求める
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def crop_contours(image, contours, output_size):
    cropped_images = []
    for contour in contours:
        # 各輪郭の中心座標を取得
        cx, cy = get_contour_center(contour)
        #　中心座標が画像の中心から離れているものを除外
        if cx > 400 and cx < 2000 and cy > 400 and cy < 2000:
            # 切り抜く範囲を計算
            x1 = max(0, cx - output_size[0] // 2)
            y1 = max(0, cy - output_size[1] // 2)
            x2 = min(image.shape[1], cx + output_size[0] // 2)
            y2 = min(image.shape[0], cy + output_size[1] // 2)
            # 画像を切り抜く
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)
    return cropped_images