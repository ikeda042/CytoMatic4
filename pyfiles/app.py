import tkinter as tk
import sqlite3
from tkinter import ttk
from PIL import Image, ImageTk
import os 

def custom_sort_key(item):
    # Fの後の数字を取得して、数値として解釈する
    f_part, c_part = item.split('F')[1].split('C')
    return (int(f_part), int(c_part))

class ImageLabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling App")
        self.image_id = "F0C0"

        self.conn = sqlite3.connect("image_labels.db")
        self.cursor = self.conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY,
                image_id TEXT,
                label TEXT
            )
        ''')
        self.conn.commit()

        self.image_ids = sorted([i.split(".")[0] for i in os.listdir("TempData/app_data")], key=custom_sort_key) 
        self.current_index = 0

        self.label_var = tk.StringVar()
        self.label_var.set("N/A") 

        label_frame = ttk.LabelFrame(self.root, text="Select Label")
        label_frame.pack(padx=10, pady=10, fill="x")

        labels = ["N/A", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
        for label in labels:
            ttk.Radiobutton(label_frame, text=label, variable=self.label_var, value=label).pack(side="left", padx=5)
        self.index_label = tk.Label(self.root, text="Image 1 of {}".format(len(self.image_ids)))
        self.index_label.pack()

        self.label_image = tk.Label(self.root)  # ここでLabelを作成
        self.label_image.pack()

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        prev_button = ttk.Button(button_frame, text="Previous", command=self.prev_image)
        prev_button.pack(side="left", padx=10)
        next_button = ttk.Button(button_frame, text="Next", command=self.next_image)
        next_button.pack(side="right", padx=10)

        submit_button = ttk.Button(self.root, text="Submit", command=self.submit_label)
        submit_button.pack(pady=10)
        root.bind('<Key>', self.key_press)

        self.load_image()  # load_imageを最初に呼び出す

    def load_image(self):
        image_id = self.image_ids[self.current_index]
        self.image_id = image_id
        image_path = f"TempData/app_data/{image_id}.png"
        img = Image.open(image_path)

        aspect_ratio = img.width / img.height
        new_width = min(800, img.width)
        new_height = int(new_width / aspect_ratio)
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        self.photo = ImageTk.PhotoImage(img)

        self.label_image.config(image=self.photo)  # label_imageを更新
        self.label_image.pack()

        self.cursor.execute("SELECT label FROM labels WHERE image_id=?", (image_id,))
        result = self.cursor.fetchone()
        if result:
            self.label_var.set(result[0])
        else:
            self.label_var.set("N/A")
        self.update_index_label()

    def update_index_label(self):
        self.index_label.config(text="Image {} of {}, cell id {}".format(self.current_index + 1, len(self.image_ids),self.image_id))

    def key_press(self, event):
        key = event.char

        if key == 'n':
            self.label_var.set('N/A')
        elif key.isdigit() and 0 <= int(key) <= 8:
            self.label_var.set(key)

        if key == '\r' or key == '\n':
            self.submit_label()
        elif event.keysym == 'Return':
            self.submit_label()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        if self.current_index < len(self.image_ids) - 1:
            self.current_index += 1
            self.load_image()

    def submit_label(self):
        image_id = self.image_ids[self.current_index]
        label = self.label_var.get()
        self.next_image()

        self.cursor.execute("SELECT * FROM labels WHERE image_id=?", (image_id,))
        result = self.cursor.fetchone()

        if result:
            self.cursor.execute("UPDATE labels SET label=? WHERE image_id=?", (label, image_id))
        else:
            self.cursor.execute("INSERT INTO labels (image_id, label) VALUES (?, ?)", (image_id, label))

        self.conn.commit()

def app() -> None:
    root = tk.Tk()
    app = ImageLabelApp(root)
    root.mainloop()


