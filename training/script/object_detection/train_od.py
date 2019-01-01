"""
Retrain the YOLO model for your own dataset.
"""
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from collections import namedtuple
import configparser
import datetime
import shutil
import os


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
    #    num_anchors//3, num_classes+5)) for l in range(3)]
    # prepare small, medium, big bounding boxes true labels.
    y_true = []
    small_object_h, medium_object_h, big_object_h = h //32, h//16, h//8
    small_object_w, medium_object_w, big_object_w = w //32, w//16, w//8

    # preparing anchor number for each size.
    # もし、smallとbigだけでいいやってなったら、これは//2にする。inputのanchorの数で調整が必要。k-meansの結果で調整するのが良いと思う。
    each_anchors = num_anchors//3

    # preparing output dimensions = [class_1_confidence, class_2_confidence, ...., h, w, x, y, loc_confidence]
    output_dims = num_classes+5
    y_true.append(Input(shape=(small_object_h, small_object_w, each_anchors, output_dims)))
    y_true.append(Input(shape=(medium_object_h, medium_object_w, each_anchors, output_dims)))
    y_true.append(Input(shape=(big_object_h, big_object_w, each_anchors, output_dims)))

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def setting_conf(conf_file):
    TrainConfig = namedtuple('TrainConfig', 'batch_size epochs input_shape num_classes ' +
                             'annotation_file classes_file anchors_file pre_train_model save_experiment_dir ' +
                             'save_model_dir save_log_dir tiny_flg freeze_body')
    config = configparser.ConfigParser()
    config.read(conf_file)

    image_height = int(config.get('image_info', 'image_height'))
    image_width = int(config.get('image_info', 'image_width'))
    image_channel_dim = int(config.get('image_info', 'image_channel_dim'))
    input_shape = (image_height, image_width, image_channel_dim)

    batch_size = int(config.get('train_info', 'batch_size'))
    epochs = int(config.get('train_info', 'epochs'))
    num_classes = int(config.get('label_info', 'num_classes'))

    now = datetime.datetime.now()
    experiment_id = now.strftime('%Y%m%d_%H%M')

    base_dir = config.get('base_info', 'base_dir')
    save_dir = base_dir + config.get('other_info', 'save_dir')
    save_experiment_dir = save_dir + experiment_id + '/'
    save_log_dir = save_experiment_dir + '/tensorboard'
    save_model_dir = save_experiment_dir + '/model/'
    pre_train_model = base_dir + config.get('train_info', 'pre_train_model')
    anchors_file = base_dir + config.get('train_info', 'anchors_file')
    annotation_file = base_dir + config.get('label_info', 'annotation_file')
    classes_file = base_dir + config.get('label_info', 'classes_file')
    tiny_flg = int(config.get('other_info', 'tiny_flg'))
    freeze_body = int(config.get('other_info', 'freeze_body')) # freeze_body in [1, 2] 1 = first 185 layer freeze 2 = first 249 layer freeze

    if not os.path.exists(save_experiment_dir):
        os.mkdir(save_experiment_dir)
    os.mkdir(save_log_dir)
    os.mkdir(save_model_dir)

    t_conf = TrainConfig(
        batch_size,
        epochs,
        input_shape,
        num_classes,
        annotation_file,
        classes_file,
        anchors_file,
        pre_train_model,
        save_experiment_dir,
        save_model_dir,
        save_log_dir,
        tiny_flg,
        freeze_body
    )

    return t_conf

def main():
    # config setting
    conf_file = '../../conf/object_detection/config_od.ini'
    t_conf = setting_conf(conf_file)

    annotation_path = t_conf.annotation_file
    log_dir = t_conf.save_log_dir
    classes_file = t_conf.classes_file
    anchors_file = t_conf.anchors_file
    num_classes = t_conf.num_classes
    input_shape = t_conf.input_shape[:2] # multiple of 32, hw
    pre_train_model = t_conf.pre_train_model
    freeze_body = t_conf.freeze_body
    save_log_dir = t_conf.save_log_dir
    save_model_dir = t_conf.save_model_dir
    epochs = t_conf.epochs
    batch_size = t_conf.batch_size

    anchors = get_anchors(anchors_file)

    shutil.copyfile(conf_file, t_conf.save_experiment_dir + conf_file.split('/')[-1])

    if t_conf.tiny_flg == 1:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=freeze_body, weights_path=pre_train_model)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=freeze_body, weights_path=pre_train_model)

    logging = TensorBoard(log_dir=save_log_dir)
    checkpoint = ModelCheckpoint(save_model_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    import pickle
    with open(t_conf.save_experiment_dir + 'vals.pkl', mode='wb') as f:
        pickle.dump(lines[num_train:], f)

    freeze_pre_train = True

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if freeze_pre_train == False:
        freeze_epoch = int(epochs/2)
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = batch_size
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=freeze_epoch,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(save_model_dir + 'trained_weights_stage_1.h5')
    else:
        freeze_model_path = '/home/yusuke/work/od_work/training/saved/object_detection/20181231_1223_960_640_200/model/trained_weights_stage_1.h5'
        freeze_epoch = 0
        model.load_weights(freeze_model_path)

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = batch_size # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=int(epochs),
            initial_epoch=freeze_epoch,
            callbacks=[logging, checkpoint, reduce_lr])
            # callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(save_model_dir + 'trained_weights_final.h5')


if __name__ == '__main__':
    main()
