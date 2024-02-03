# ============== IMPORTANT - USE YoloV8 environment ====================
# ============== IMPORTANT - USE YoloV8 environment ====================

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np
from ultralytics import YOLO

class App:
    def __init__(self, master, model_path, folder_path):
        self.master = master
        self.master.title("Image Viewer")
        self.master.geometry("660x800")

        self.model = YOLO(model_path)
        self.folder_path = folder_path

        self.photo = None

        # get a list of all the files in the folder
        self.file_list = os.listdir(self.folder_path)

        # create a canvas to display the image
        self.canvas = tk.Canvas(self.master, width=640, height=640)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky='we')

        # Create a button to select the model path
        self.select_model_button = tk.Button(self.master, text="Select Model", command=self.select_model_path)
        self.select_model_button.grid(row=1, column=0, padx=10, pady=10, sticky='we')

        # Create a button to select the image folder path
        self.select_folder_button = tk.Button(self.master, text="Select tes images folder", command=self.select_folder_path)
        self.select_folder_button.grid(row=2, column=0, padx=10, pady=10, sticky='we')

        # set the current image index to 0
        self.image_index = 0

        # create a button to go to the next image
        self.next_button = tk.Button(self.master, text="Start predicting", command=self.next_image)
        self.next_button.grid(row=3, column=0, padx=10, pady=10, sticky='we')

    def select_model_path(self):
        # Open a file dialog to select the model path
        model_path = filedialog.askopenfilename()

        # Load the YOLO model with the selected path
        self.model = YOLO(model_path)

    def select_folder_path(self):
        # Open a file dialog to select the folder path
        self.folder_path = filedialog.askdirectory()
        # get a list of all the files in the folder
        self.file_list = os.listdir(self.folder_path)

    def display_image(self):
        # joint the folder path and the image name
        image_path = os.path.join(self.folder_path, self.file_list[self.image_index])

        # Run inference on a single image
        results = self.model(image_path)

        # get the label, confidence and xyxy from the results
        for r in results:
            label = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)

            # joint the label, confidence and xyxy into a single list of one dimension
            label_conf_xyxy = np.concatenate((label.reshape(-1,1), conf.reshape(-1,1), xyxy), axis=1)

        # get p1 and p2
        p1 = label_conf_xyxy[:,2:4]
        p2 = label_conf_xyxy[:,4:6]

        # convert p1 and p2 to a list of tuples adn keep them as integers
        p1 = [tuple(map(int, p)) for p in p1]
        p2 = [tuple(map(int, p)) for p in p2]

        # open the image and convert it to a tkinter PhotoImage
        img = Image.open(image_path)
        img = img.convert('RGB')

        # draw the bounding boxes on the image
        draw = ImageDraw.Draw(img)
        for i in range(len(p1)):
            draw.rectangle((p1[i], p2[i]), outline=(124,255,0), width=2)

        # convert the image to a tkinter PhotoImage and add it to the canvas
        self.photo = ImageTk.PhotoImage(img)

        # create a white canvas the size of the image
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        
        # update the image on the canvas
        self.canvas.image = self.photo
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky='we')
        
        # create a button to go to the previous image
        self.prev_button = tk.Button(self.master, text="Previous Image", command=self.previous_image)
        self.prev_button.grid(row=1, column=0, padx=10, pady=10, sticky='we')

        # create a button to go to the next image
        self.next_button = tk.Button(self.master, text="Next Image", command=self.next_image)
        self.next_button.grid(row=2, column=0, padx=10, pady=10, sticky='we')


    def next_image(self):
        # Remove the "Start predicting" button from the grid
        self.next_button.grid_forget()
        # Reduce the size of the window
        self.master.geometry("660x780")
        # increment the image index
        self.image_index += 1

        # if we've reached the end of the file list, start over
        if self.image_index == len(self.file_list):
            self.image_index = 0

        # clear the canvas
        self.canvas.delete("all")

        # display the next image
        self.display_image()

    def previous_image(self):
        # decrement the image index
        self.image_index -= 1

        # if we've reached the beginning of the file list, go to the end
        if self.image_index < 0:
            self.image_index = len(self.file_list) - 1

        # clear the canvas
        self.canvas.delete("all")

        # display the previous image
        self.display_image()

# Define the default model path
default_model_path = 'runs/detect/train3/weights/best.pt'
# Define the default folder path
folder_path = 'dataset/images/val'

root = tk.Tk()
app = App(root, default_model_path, folder_path)
root.mainloop()