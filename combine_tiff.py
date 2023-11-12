from PIL import Image

def combine_tiff_layers(file_a, file_b, output_file):
    # ファイルを開く
    with Image.open(file_a) as img_a, Image.open(file_b) as img_b:
        # 出力画像リスト
        output_images = []

        # レイヤー数を取得（nは両ファイルで同じと仮定）
        n = img_a.n_frames // 2

        for i in range(n):
            # ファイルAの位相差画像を取得
            img_a.seek(i * 2)  # 0, 2, 4, ...のレイヤー
            output_images.append(img_a.copy())

            # ファイルAの蛍光画像を取得
            img_a.seek(i * 2 + 1)  # 1, 3, 5, ...のレイヤー
            output_images.append(img_a.copy())

            # ファイルBの発光画像を取得
            img_b.seek(i * 2 + 1)  # 1, 3, 5, ...のレイヤー
            output_images.append(img_b.copy())

        # 新しいTIFFファイルを作成
        output_images[0].save(output_file, save_all=True, append_images=output_images[1:])


# 実行
combine_tiff_layers('sk326tri120min.tif', 'sk326tri120minpi.tif', 'data.tif')


