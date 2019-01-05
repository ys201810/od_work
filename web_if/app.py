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
# app.config['RESIZE_FOLDER'] = UPLOAD_FOLDER + '/resize_images'
# app.config['RESULT_TINY_FOLDER'] = UPLOAD_FOLDER + '/result_images/tiny_yolo'
# app.config['RESULT_V3_FOLDER'] = UPLOAD_FOLDER + '/result_images/yolov3'
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MODEL_DIR'] = './model/'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/det_andon_tiny_rand_960_640')
def det_andon_tiny_rand_960_640():
    return render_template('detection_andon_tiny_960_640.html', model_name = 'Tiny YOLO(image_size:960_640)', scores = [round((i +1) * 0.1, 1) for i in range(10)])

@app.route('/det_andon_v3_rand_960_640')
def det_andon_v3_rand_960_640():
    return render_template('detection_andon_yolov3_960_640.html', model_name = 'YOLO v3(image_size:960_640)', scores = [round((i +1) * 0.1, 1) for i in range(10)])

@app.route('/det_andon_tiny_rand_480_320')
def det_andon_tiny_rand_480_320():
    return render_template('detection_andon_tiny_480_320.html', model_name = 'Tiny YOLO(image_size:480_320)', scores = [round((i +1) * 0.1, 1) for i in range(10)])

@app.route('/det_andon_v3_rand_480_320')
def det_andon_v3_rand_480_320():
    return render_template('detection_andon_yolov3_480_320.html', model_name = 'YOLO v3(image_size:480_320)', scores = [round((i +1) * 0.1, 1) for i in range(10)])

@app.route('/send_andon_det_tiny_yolo_960_640', methods=['GET', 'POST'])
def send_andon_det_tiny_yolo_960_640():
    model_kind = 'tiny_yolo_960_640'
    use_model = './model/tiny_trained_weights_960_640_final.h5'
    anchors_path = 'model/tiny_yolo_anchors_andon_train_960_640.txt'
    resize_size = (960, 640)
    selected_score = float(request.form['score'].split('_')[1]) # score's value format = 'score_0.1' so pick up after _.
    print(selected_score)
    img_url, result_url, elapse_time = det_inference(model_kind, use_model, anchors_path, selected_score, resize_size)
    return render_template('detection_andon_tiny_960_640.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'Tiny YOLO(image_size:960_640)', scores = [round((i +1) * 0.1, 1) for i in range(10)], selected_score=selected_score)

@app.route('/send_andon_det_yolov3_960_640', methods=['GET', 'POST'])
def send_andon_det_yolov3_960_640():
    model_kind = 'yolov3_960_640'
    use_model = './model/trained_weights_960_640_yolov3_final.h5'
    anchors_path = './model/yolov3_anchors_andon_train_960_640.txt'
    resize_size = (960, 640)
    selected_score = float(request.form['score'].split('_')[1]) # score's value format = 'score_0.1' so pick up after _.
    print(selected_score)
    img_url, result_url, elapse_time = det_inference(model_kind, use_model, anchors_path, selected_score, resize_size)
    return render_template('detection_andon_yolov3_960_640.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'YOLO v3(image_size:960_640)', scores = [round((i +1) * 0.1, 1) for i in range(10)], selected_score=selected_score)

@app.route('/send_andon_det_tiny_yolo_480_320', methods=['GET', 'POST'])
def send_andon_det_tiny_yolo_480_320():
    model_kind = 'tiny_yolo_960_640'
    use_model = './model/tiny_trained_weights_480_320_final.h5'
    anchors_path = 'model/tiny_yolo_anchors_andon_train_480_320.txt'
    resize_size = (480, 320)
    selected_score = float(request.form['score'].split('_')[1]) # score's value format = 'score_0.1' so pick up after _.
    print(selected_score)
    img_url, result_url, elapse_time = det_inference(model_kind, use_model, anchors_path, selected_score, resize_size)
    return render_template('detection_andon_tiny_480_320.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'Tiny YOLO(image_size:480_320)', scores = [round((i +1) * 0.1, 1) for i in range(10)], selected_score=selected_score)

@app.route('/send_andon_det_yolov3_480_320', methods=['GET', 'POST'])
def send_andon_det_yolov3_480_320():
    model_kind = 'yolov3_480_320'
    use_model = './model/trained_weights_480_320_yolov3_final.h5'
    anchors_path = './model/yolov3_anchors_andon_train_480_320.txt'
    resize_size = (480, 320)
    selected_score = float(request.form['score'].split('_')[1]) # score's value format = 'score_0.1' so pick up after _.
    print(selected_score)
    img_url, result_url, elapse_time = det_inference(model_kind, use_model, anchors_path, selected_score, resize_size)
    return render_template('detection_andon_yolov3_480_320.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'YOLO v3(image_size:480_320)', scores = [round((i +1) * 0.1, 1) for i in range(10)], selected_score=selected_score)


def det_inference(model_kind, use_model, anchors_path, score, resize_size):
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            os.chdir(os.path.dirname(__file__))
            print(filename)

            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resize_img = cv2.resize(img, resize_size)
            resize_name = os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0] + '_' +
                                       str(resize_size[0]) + '_' + str(resize_size[1]) + '.' + filename.split('.')[1])
            cv2.imwrite(resize_name, resize_img)
            resize_img_url = '/uploads/' + filename.split('.')[0] + '_' + str(resize_size[0]) + '_' + \
                             str(resize_size[1]) + '.' + filename.split('.')[1]

            image = Image.open(resize_name)
            yolo = YOLO(model_path=use_model, anchors_path=anchors_path, score=score)
            r_image, elapse_time = yolo.detect_image(image)

            if model_kind.find('yolov3') > 0:
                result_name = os.path.join(app.config['UPLOAD_FOLDER'],
                                         filename.split('.')[0] + '_' + str(score).replace('.', '') + '_' + '_'.join(map(str,list(resize_size))) + '_result_yolov3.' + filename.split('.')[1])
                result_img_url = '/uploads/' + filename.split('.')[0] + '_' + str(score).replace('.', '') + '_' +  '_'.join(map(str,list(resize_size))) + '_result_yolov3.' + filename.split('.')[1]
            else:
                result_name = os.path.join(app.config['UPLOAD_FOLDER'],
                                         filename.split('.')[0] + '_' + str(score).replace('.', '') + '_' + '_'.join(map(str,list(resize_size))) + '_result_tiny_yolo.' + filename.split('.')[1])
                result_img_url = '/uploads/' + filename.split('.')[0] + '_' + str(score).replace('.', '') + '_' + '_'.join(map(str,list(resize_size))) + '_result_tiny_yolo.' + filename.split('.')[1]

            r_image.save(result_name, quality=100, optimize=True)
            backend.clear_session()

            return resize_img_url, result_img_url, elapse_time


@app.route('/uploads/<filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

"""
@app.route('/uploads/resize_images/<filename>')
def resize_images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'] + 'resize_images/', filename)

@app.route('/uploads/result_images/tiny_yolo/<filename>')
def result_tiny_yolo(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'] + 'result_images/tiny_yolo/', filename)

@app.route('/uploads/result_images/yolov3/<filename>')
def result_yolov3(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'] + 'result_images/yolov3/', filename)
"""

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
