# # ============== IMPORTANT - USE YoloV8 environment ====================
# # ============== IMPORTANT - USE YoloV8 environment ====================


from ultralytics import YOLO
import os
from PIL import Image

# Load a pretrained YOLO model (recommended for training)
model = YOLO('runs/detect/train3/weights/best.pt')

# specify the path to the folder containing the test images
folder_path = 'dataset/images/val'

# get a list of all the files in the folder
file_list = os.listdir(folder_path)

# PRINT THE FIRST 10 IMAGES
image = file_list[2]

# Run inference on a single image
results = model.predict('dataset/images/val/'+image, save=False, show=False)

# Show the results with openCV
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # show the image using OpenCV
    im.show()
    # keep the window open until we press a key
    cv2.waitKey(0)
