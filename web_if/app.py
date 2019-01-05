# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename
import cv2
from od_yolo import YOLO
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
    """アップロード画像の拡張子チェック"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """access point/配下の挙動記述"""
    return render_template('index.html')

@app.route('/det_andon_tiny_rand_960_640')
def det_andon_tiny_rand_960_640():
    """960*640のtiny yoloでの解析ページの初期状態記述"""
    return render_template('detection_andon_tiny_960_640.html', model_name = 'Tiny YOLO(image_size:960_640)',
                           scores = [round((i +1) * 0.1, 1) for i in range(10)])

@app.route('/det_andon_v3_rand_960_640')
def det_andon_v3_rand_960_640():
    """960*640のyolov3での解析ページの初期状態記述"""
    return render_template('detection_andon_yolov3_960_640.html', model_name = 'YOLO v3(image_size:960_640)',
                           scores = [round((i +1) * 0.1, 1) for i in range(10)])

@app.route('/det_andon_tiny_rand_480_320')
def det_andon_tiny_rand_480_320():
    """480*320のtiny yoloでの解析ページの初期状態記述"""
    return render_template('detection_andon_tiny_480_320.html', model_name = 'Tiny YOLO(image_size:480_320)',
                           scores = [round((i +1) * 0.1, 1) for i in range(10)])

@app.route('/det_andon_v3_rand_480_320')
def det_andon_v3_rand_480_320():
    """480*320のyolov3での解析ページの初期状態記述"""
    return render_template('detection_andon_yolov3_480_320.html', model_name = 'YOLO v3(image_size:480_320)',
                           scores = [round((i +1) * 0.1, 1) for i in range(10)])

@app.route('/send_andon_det_tiny_yolo_960_640', methods=['GET', 'POST'])
def send_andon_det_tiny_yolo_960_640():
    """960*480のtiny yoloでの画像アップロード後の解析と結果表示記述"""
    model_kind = 'tiny_yolo_960_640'
    use_model = './model/tiny_trained_weights_960_640_final.h5'
    anchors_path = 'model/tiny_yolo_anchors_andon_train_960_640.txt'
    resize_size = (960, 640)
    selected_score = float(request.form['score'].split('_')[1]) # score's value format = 'score_0.1' so pick up after _.
    print(selected_score)
    img_url, result_url, elapse_time = det_inference(model_kind, use_model, anchors_path, selected_score, resize_size)
    return render_template('detection_andon_tiny_960_640.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'Tiny YOLO(image_size:960_640)',
                           scores = [round((i +1) * 0.1, 1) for i in range(10)], selected_score=selected_score)

@app.route('/send_andon_det_yolov3_960_640', methods=['GET', 'POST'])
def send_andon_det_yolov3_960_640():
    """960*640のyolov3での画像アップロード後の解析と結果表示記述"""
    model_kind = 'yolov3_960_640'
    use_model = './model/trained_weights_960_640_yolov3_final.h5'
    anchors_path = './model/yolov3_anchors_andon_train_960_640.txt'
    resize_size = (960, 640)
    selected_score = float(request.form['score'].split('_')[1]) # score's value format = 'score_0.1' so pick up after _.
    print(selected_score)
    img_url, result_url, elapse_time = det_inference(model_kind, use_model, anchors_path, selected_score, resize_size)
    return render_template('detection_andon_yolov3_960_640.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'YOLO v3(image_size:960_640)',
                           scores = [round((i +1) * 0.1, 1) for i in range(10)], selected_score=selected_score)

@app.route('/send_andon_det_tiny_yolo_480_320', methods=['GET', 'POST'])
def send_andon_det_tiny_yolo_480_320():
    """480*320のtiny yoloでの画像アップロード後の解析と結果表示記述"""
    model_kind = 'tiny_yolo_960_640'
    use_model = './model/tiny_trained_weights_480_320_final.h5'
    anchors_path = 'model/tiny_yolo_anchors_andon_train_480_320.txt'
    resize_size = (480, 320)
    selected_score = float(request.form['score'].split('_')[1]) # score's value format = 'score_0.1' so pick up after _.
    print(selected_score)
    img_url, result_url, elapse_time = det_inference(model_kind, use_model, anchors_path, selected_score, resize_size)
    return render_template('detection_andon_tiny_480_320.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'Tiny YOLO(image_size:480_320)',
                           scores = [round((i +1) * 0.1, 1) for i in range(10)], selected_score=selected_score)

@app.route('/send_andon_det_yolov3_480_320', methods=['GET', 'POST'])
def send_andon_det_yolov3_480_320():
    """480*320のyolo v3での画像アップロード後の解析と結果表示記述"""
    model_kind = 'yolov3_480_320'
    use_model = './model/trained_weights_480_320_yolov3_final.h5'
    anchors_path = './model/yolov3_anchors_andon_train_480_320.txt'
    resize_size = (480, 320)
    selected_score = float(request.form['score'].split('_')[1]) # score's value format = 'score_0.1' so pick up after _.
    print(selected_score)
    img_url, result_url, elapse_time = det_inference(model_kind, use_model, anchors_path, selected_score, resize_size)
    return render_template('detection_andon_yolov3_480_320.html', img_url=img_url, result_url=result_url, elapse_time=round(elapse_time, 2),
                           model_name = 'YOLO v3(image_size:480_320)',
                           scores = [round((i +1) * 0.1, 1) for i in range(10)], selected_score=selected_score)


def det_inference(model_kind, use_model, anchors_path, score, resize_size):
    """
    アップロードされた画像を用いた解析の実行と結果のreturn
    Parameters
    ----------
    model_kind: モデルの種類
    use_model: モデルファイルのパス
    anchors_path: アンカーファイルのパス
    score: 結果表示の閾値
    resize_size: 画像サイズ
    """
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            org_img_name = secure_filename(img_file.filename)
            img_name = org_img_name.split('.')[0]
            img_postfix = org_img_name.split('.')[1]
            os.chdir(os.path.dirname(__file__))
            # アップロード画像の保存
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], org_img_name))
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], org_img_name))

            # リサイズ画像の保存
            resize_img = cv2.resize(img, resize_size)
            resize_img_name = img_name + '_' + '_'.join(map(str,list(resize_size))) + '_resize.' + img_postfix
            resize_name = os.path.join(app.config['UPLOAD_FOLDER'], resize_img_name)
            cv2.imwrite(resize_name, resize_img)
            resize_img_url = '/uploads/' + resize_img_name

            # リサイズ画像を用いた解析の実行と解析結果の保存
            image = Image.open(resize_name)
            yolo = YOLO(model_path=use_model, anchors_path=anchors_path, score=score)
            r_image, elapse_time = yolo.detect_image(image)

            if model_kind.find('yolov3') != -1:
                # yolov3の場合
                result_img_name = img_name + '_' + str(score).replace('.', '') + '_' + \
                                  '_'.join(map(str,list(resize_size))) + '_result_yolov3.' + img_postfix
                result_name = os.path.join(app.config['UPLOAD_FOLDER'], result_img_name)
                result_img_url = '/uploads/' + result_img_name
            else:
                result_img_name = img_name + '_' + str(score).replace('.', '') + '_' + \
                                  '_'.join(map(str,list(resize_size))) + '_result_tiny_yolo.' + img_postfix
                result_name = os.path.join(app.config['UPLOAD_FOLDER'], result_img_name)
                result_img_url = '/uploads/' + result_img_name

            r_image.save(result_name, quality=100, optimize=True)
            backend.clear_session()

            return resize_img_url, result_img_url, elapse_time


@app.route('/uploads/<filename>')
def uploads(filename):
    """./upload/にファイルが格納された時に、これをHTMLで表示できるようにしておく関数"""
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
