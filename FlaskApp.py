# create a flask server to load images from a folder and display them on a webpage
# use a html template to display the image file list and the images

# ============== IMPORTANT - USE YoloV8 environment ====================
# ============== IMPORTANT - USE YoloV8 environment ====================

# TODO: add its own python environment

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw
import os
import numpy as np
from ultralytics import YOLO
import os
import glob


# load the YOLO model from a given path
model_path = 'models/best.pt'

model = YOLO(model_path)

app = Flask(__name__)

# define the path to the image folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')
APP_IMAGES = os.path.join(APP_STATIC, 'images')
APP_TEMP = os.path.join(APP_STATIC, 'temp')

# delete all files in the temp folder
files = glob.glob(APP_TEMP + '/*')
for f in files:
    os.remove(f)

# define the path to the template folder
APP_TEMPLATES = os.path.join(APP_STATIC, 'templates')

# define the function to predict objects in an image
def predict_objects(image_path):
    # make the prediction with the YOLO model
    results = model(image_path)
    # return the prediction results
    return results

# define the route to the index page
@app.route('/')
def index():
    # get the list of image files
    image_names = os.listdir(APP_IMAGES)
    # get the name of the current image
    current_image = request.args.get('image_name')
    
    # predict objects in the current image
    if current_image is not None:
        image_path = os.path.join(APP_IMAGES, current_image)
        results = predict_objects(image_path)
        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

            # save the image to a temporary file
            temp_file = f"static/temp/{current_image}"
            im.save(temp_file)

    else:
        results = None
    return render_template('index.html', image_names=image_names, current_image=current_image)


# define the route to the image file
@app.route('/images/<image_name>')
def images(image_name):
    # return the image file
    return send_from_directory(APP_IMAGES, image_name)


# define the route to the upload page
@app.route('/upload')
def upload():
    # render the upload.html template
    return render_template("upload.html")


# define the route to the upload action
@app.route('/uploadAction', methods=['POST'])
def uploadAction():
    # get the file from the request
    file = request.files['file']
    # save the file to the image folder
    file.save(os.path.join(APP_IMAGES, file.filename))
    # redirect to the index page
    return redirect(url_for('index'))


# run the app
if __name__ == '__main__':
    app.run(debug=True)

# add close app when the server is stopped


