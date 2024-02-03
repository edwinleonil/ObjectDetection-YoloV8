# This scripts runs with YoloV8 environment
# The app takes the model path and the test images folder path as input

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
from PIL import ImageDraw, ImageFont

# TODO: Improve model performance by improving bounding box annotations
class ImagePredictor:
    def __init__(self, model_path, base_path, subfolders, label_txt_path):
        self.model = YOLO(model_path)
        self.base_path = base_path
        self.subfolders = subfolders
        self.label_txt_path = label_txt_path
        self.folder_path = os.path.join(base_path, subfolders[0])  # Default to first subfolder
        self.file_list = os.listdir(self.folder_path)
        self.root = tk.Tk()
        self.root.geometry('660x700')  # Set the window size
        self.entry_label = tk.Label(self.root, text="Image File")
        self.entry = ttk.Combobox(self.root, values=self.file_list, width=40)
        self.entry.bind("<<ComboboxSelected>>", self.update_image)
        self.image_label = tk.Label(self.root)
        self.combobox_label = tk.Label(self.root, text="Image Test Folder")
        self.combobox = ttk.Combobox(self.root, values=self.subfolders)
        self.combobox.bind("<<ComboboxSelected>>", self.update_folder_path)
        self.combobox.current(0)  # Default selection

    def load_and_predict(self, file_name):
        image_path = os.path.join(self.folder_path, file_name)
        resize_image = Image.open(image_path).resize((640, 640))
        results = self.model.predict(resize_image, save=False, show=False)
        for r in results:
            im_array = r.plot()
            img = Image.fromarray(im_array[..., ::-1])
        return img

    def get_bounding_boxes_from_txt(self, file_name):
        txt_file_path = os.path.join(self.label_txt_path, file_name.replace('.png', '.txt'))
        bounding_boxes = []
        class_names = ['A', 'B', 'C', 'D', 'E', 'F']
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    values = line.strip().split()
                    class_index = int(values[0])
                    class_name = class_names[class_index]
                    x, y, w, h = map(float, values[1:])
                    x_min = int((x - w / 2) * 640)
                    y_min = int((y - h / 2) * 640)
                    x_max = int((x + w / 2) * 640)
                    y_max = int((y + h / 2) * 640)
                    bounding_boxes.append((x_min, y_min, x_max, y_max, class_name))  # Add class_name to each bounding box
        return bounding_boxes

    def update_image(self, event=None):
        file_name = self.entry.get()
        image = self.load_and_predict(file_name)

        # Draw predicted bounding box on the image
        draw = ImageDraw.Draw(image)

        # Get bounding boxes from txt file
        bounding_boxes = self.get_bounding_boxes_from_txt(file_name)
        for bbox in bounding_boxes:
            draw.rectangle(bbox[:4], outline="blue", width=2)
            # add the class name on the bottom right corner of the bounding box
            draw.text((bbox[2], bbox[3]-12), bbox[4], fill="white", font=ImageFont.truetype("arial", 12), stroke_fill='green', stroke_width=2)  # Use bbox[4] as the class name

        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def update_folder_path(self, event):
        selected_subfolder = self.combobox.get()
        self.folder_path = os.path.join(self.base_path, selected_subfolder)
        self.file_list = os.listdir(self.folder_path)
        self.entry['values'] = self.file_list
        self.entry.current(0)  # Default to first file in the updated list
        self.update_image()

    def run(self):
        self.root.grid_columnconfigure(0, weight=1)  # Allow column to expand
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

        self.combobox_label.grid(row=0, column=0, sticky="nsew", pady=(10, 10))
        self.combobox.grid(row=0, column=1, sticky="nsew", pady=(10, 10))
        self.entry_label.grid(row=0, column=2, sticky="nsew", pady=(10, 10))
        self.entry.grid(row=0, column=3, sticky="nsew", pady=(10, 10))
        self.image_label.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(10, 10))

        self.root.mainloop()

model_path = 'runs/detect/train5/weights/best.pt'
test_folder_path = 'dataset/images'
label_txt_path = 'dataset/labels/train'
list_subfolders = os.listdir(test_folder_path)
predictor = ImagePredictor(model_path, test_folder_path, list_subfolders, label_txt_path)
predictor.run()