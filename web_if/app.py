# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import cv2
import numpy as np
from od_predict import YOLO
from PIL import Image
import os
from keras.backend import tensorflow_backend as backend


app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg', 'PNG', 'JPG', 'GIF', 'JPEG'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MODEL_DIR'] = './model/'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    return render_template('index.html')

@app.route('/detec_sf_rand')
def detec_sf_rand():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    return render_template('detection_andon.html', model_name = 'tiny_960_640_v1')

@app.route('/send_sf_det', methods=['GET', 'POST'])
def send_sf_det():
    # use_model = './model/trained_weights_960_640_yolov3_final.h5'
    use_model = './model/tiny_trained_weights_960_640_final.h5'
    img_url, result_url, elapse_time = det_inference(use_model)
    return render_template('detection_andon.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'tiny_960_640_v1')

def det_inference(use_model):
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            os.chdir(os.path.dirname(__file__))
            print(filename)

            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resize_img = cv2.resize(img, (960, 640))
            resize_name = os.path.join(app.config['UPLOAD_FOLDER'],
                                     filename.split('.')[0] + '_resize.' + filename.split('.')[1])
            cv2.imwrite(resize_name, resize_img)
            resize_img_url = '/uploads/' + filename.split('.')[0] + '_resize.' + filename.split('.')[1]

            image = Image.open(resize_name)
            yolo = YOLO()
            r_image, elapse_time = yolo.detect_image(image)

            result_name = os.path.join(app.config['UPLOAD_FOLDER'],
                                     filename.split('.')[0] + '_result_det.' + filename.split('.')[1])
            r_image.save(result_name, quality=100, optimize=True)
            # cv2.imwrite(result_name, r_image)
            result_img_url = '/uploads/' + filename.split('.')[0] + '_result_det.' + filename.split('.')[1]

            backend.clear_session()
            return resize_img_url, result_img_url, elapse_time


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
