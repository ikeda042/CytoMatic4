import tkinter as tk
from tkinter import filedialog
from main import main
import os

def PhenoPixel4():
    def run_main():
        file_name = file_name_var.get()
        param1 = int(param1_var.get())
        param2 = int(param2_var.get())
        img_size = int(img_size_var.get())
        mode = mode_var.get()
        layer_mode = layer_mode_var.get()
        
        with open("param_data.txt",'w') as temp_file:
            temp_file.write(f'{file_name}\n')
            temp_file.write(f'{param1}\n')
            temp_file.write(f'{param2}\n')
            temp_file.write(f'{img_size}\n')
            temp_file.write(f'{mode}\n')
            temp_file.write(f'{layer_mode}\n')
        root.destroy()
        

    # GUIウィンドウの初期化
    root = tk.Tk()
    root.geometry("400x500")
    # ウィンドウサイズの固定
    root.resizable(False, False)
    root.title("CytoMatic4")

    # ファイル選択
    file_name_var = tk.StringVar()
    tk.Button(root, text="Select File", command=lambda: file_name_var.set(filedialog.askopenfilename())).pack()

    # パラメータ入力フィールド
    tk.Label(root, text="Parameter 1").pack()
    param1_var = tk.StringVar(value="130")
    tk.Entry(root, textvariable=param1_var).pack()

    tk.Label(root, text="Parameter 2").pack()
    param2_var = tk.StringVar(value="255")
    tk.Entry(root, textvariable=param2_var).pack()

    tk.Label(root, text="Image Size").pack()
    img_size_var = tk.StringVar(value="200")
    tk.Entry(root, textvariable=img_size_var).pack()

    # ラジオボタン
    mode_var = tk.StringVar(value="all")
    tk.Label(root, text="Mode").pack()
    tk.Radiobutton(root, text="All", variable=mode_var, value="all").pack()
    tk.Radiobutton(root, text="Data Analysis", variable=mode_var, value="data_analysis").pack()
    tk.Radiobutton(root, text="Data Analysis(all db)", variable=mode_var, value="data_analysis_all").pack()
    tk.Radiobutton(root, text="Delete All", variable=mode_var, value="delete_all").pack()

    layer_mode_var = tk.StringVar(value="dual")
    tk.Label(root, text="Layer Mode").pack()
    tk.Radiobutton(root, text="Dual", variable=layer_mode_var, value="dual").pack()
    tk.Radiobutton(root, text="Single", variable=layer_mode_var, value="single").pack()
    tk.Radiobutton(root, text="Normal", variable=layer_mode_var, value="normal").pack()

    # 実行ボタン
    tk.Button(root, text="Run", command=run_main).pack()

    # GUIループの開始
    root.mainloop()


if __name__ == "__main__":
    PhenoPixel4()
    with open("param_data.txt",mode = "r") as fp:
        file_name = fp.readline().rstrip("\n")
        param1 = int(fp.readline().rstrip("\n"))
        param2 = int(fp.readline().rstrip("\n"))
        img_size = int(fp.readline().rstrip("\n"))
        mode = fp.readline().rstrip("\n")
        layer_mode = fp.readline().rstrip("\n")
    main(file_name, param1, param2, img_size, mode, layer_mode)
    try:
        os.remove("fileparam_data.txt")
    except:
        pass