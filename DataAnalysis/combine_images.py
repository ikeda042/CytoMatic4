import numpy as np
import matplotlib.pyplot as plt
import cv2



def combine_images_function(total_rows, total_cols, image_size, num_images, filename, single_layer_mode, dual_layer_mode):
    # まとめる画像のサイズを計算
    result_image = np.zeros((total_rows * image_size, total_cols * image_size, 3), dtype=np.uint8)
    num_images += 1
    # 画像をまとめる処理 Replot のフォルダから
    for i in range(total_rows):
        for j in range(total_cols):
            image_index = i * total_cols + j  # 画像のインデックス
            if image_index <num_images:
                image_path = f'Cell/replot/{image_index}.png'  # 画像のパスを適切に設定
                print(image_path)
                # 画像を読み込んでリサイズ
                img = cv2.imread(image_path)
                img = cv2.resize(img, (image_size, image_size))
                # まとめる画像に配置
                result_image[i * image_size: (i + 1) * image_size,
                            j * image_size: (j + 1) * image_size] = img

    plt.axis('off')
    # まとめた画像を保存
    cv2.imwrite(f'{filename}_replot.png', result_image)
    plt.close()


    # 画像をまとめる処理 Histoのフォルダから
    for i in range(total_rows):
        for j in range(total_cols):
            image_index = i * total_cols + j  # 画像のインデックス
            if image_index <num_images:
                image_path = f'Cell/histo/{image_index}.png'  # 画像のパスを適切に設定
                print(image_path)
                # 画像を読み込んでリサイズ
                img = cv2.imread(image_path)
                img = cv2.resize(img, (image_size, image_size))
                # まとめる画像に配置
                result_image[i * image_size: (i + 1) * image_size,
                            j * image_size: (j + 1) * image_size] = img

    plt.axis('off')
    # まとめた画像を保存
    cv2.imwrite(f'{filename}_histo.png', result_image)
    plt.close()




    # for i in range(total_rows):
    #     for j in range(total_cols):
    #         image_index = i * total_cols + j
    #         if image_index <num_images:
    #             image_path = f'Cell/replot_map/{image_index}.png'  
    #             print(image_path)
    #             img = cv2.imread(image_path)
    #             img = cv2.resize(img, (image_size, image_size))
    #             result_image[i * image_size: (i + 1) * image_size,
    #                         j * image_size: (j + 1) * image_size] = img
    # plt.axis('off')
    # cv2.imwrite(f'{filename}_replot_map.png', result_image)

    for i in range(total_rows):
        for j in range(total_cols):
            image_index = i * total_cols + j 
            if image_index <num_images:
                image_path = f'Cell/ph/{image_index}.png'  
                print(image_path)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (image_size, image_size))
                result_image[i * image_size: (i + 1) * image_size,
                            j * image_size: (j + 1) * image_size] = img
    plt.axis('off')
    cv2.imwrite(f'{filename}_ph.png', result_image)
    if not single_layer_mode:
        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j  
                if image_index <num_images:
                    image_path = f'Cell/fluo1/{image_index}.png' 
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[i * image_size: (i + 1) * image_size,
                                j * image_size: (j + 1) * image_size] = img
        plt.axis('off')
        cv2.imwrite(f'{filename}_fluo1.png', result_image)

    if dual_layer_mode:
        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j  
                if image_index <num_images:
                    image_path = f'Cell/fluo2/{image_index}.png' 
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[i * image_size: (i + 1) * image_size,
                                j * image_size: (j + 1) * image_size] = img
        plt.axis('off')
        cv2.imwrite(f'{filename}_fluo2.png', result_image)

    if False:
        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j  
                if image_index <num_images:
                    image_path = f'Cell/fluo_incide_cell_only/{image_index}.png' 
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[i * image_size: (i + 1) * image_size,
                                j * image_size: (j + 1) * image_size] = img
        plt.axis('off')
        cv2.imwrite(f'{filename}_fluo_only_inside_cell.png', result_image)


    if not single_layer_mode:
        for i in range(total_rows):
            for j in range(total_cols):
                image_index = i * total_cols + j  
                if image_index <num_images:
                    image_path = f'Cell/gradient_magnitudes/gradient_magnitude{image_index}.png' 
                    print(image_path)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (image_size, image_size))
                    result_image[i * image_size: (i + 1) * image_size,
                                j * image_size: (j + 1) * image_size] = img
        plt.axis('off')
        cv2.imwrite(f'{filename}_gradient_magnitude.png', result_image)
    
    for i in range(total_rows):
        for j in range(total_cols):
            image_index = i * total_cols + j  
            if image_index <num_images:
                image_path = f'Cell/histo_cumulative/{image_index}.png' 
                print(image_path)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (image_size, image_size))
                result_image[i * image_size: (i + 1) * image_size,
                            j * image_size: (j + 1) * image_size] = img
    plt.axis('off')
    cv2.imwrite(f'{filename}_histo_cumulative.png', result_image)




