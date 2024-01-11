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

def unify_images_ndarray6(image1, image2, image3, image4, image5, image6, output_name):
    # すべての画像の幅と高さを取得
    widths = [image1.shape[1], image2.shape[1], image3.shape[1], image4.shape[1], image5.shape[1], image6.shape[1]]
    heights = [image1.shape[0], image2.shape[0], image3.shape[0], image4.shape[0], image5.shape[0], image6.shape[0]]

    # 最大の幅と高さを決定
    max_width = max(widths)
    max_height = max(heights)

    # キャンバスのサイズを決定（3行2列）
    canvas_width = max_width * 2
    canvas_height = max_height * 3

    # キャンバスを作成
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 各画像をキャンバスに配置
    images = [image1, image2, image3, image4, image5, image6]
    for i, img in enumerate(images):
        row = i // 2
        col = i % 2
        canvas[row*max_height:(row+1)*max_height, col*max_width:(col+1)*max_width, :] = cv2.resize(img, (max_width, max_height))

    # 画像を保存
    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

# unify_images_ndarray(
#     cv2.resize(cv2.imread("ph1.png"),(600,600)),
#     cv2.imread("replot1.png"),
#     "result"  )

# unify_images_ndarray6(
#     cv2.imread("poly_reg_k5.png"),
#     cv2.imread("poly_reg_k6.png"),
#     cv2.imread("poly_reg_k7.png"),
#     cv2.imread("poly_reg_k8.png"),
#     cv2.imread("poly_reg_k9.png"),
#     cv2.imread("poly_reg_k10.png"),
#     "result_kth10"
# )