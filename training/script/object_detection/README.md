## about this directory
This directory is for cnn object detection training.
In this directory, you can use only yolov3 now. I want to add other networks in the future.

## how to use
### 1. Make a dataset.
 Prepare train and test directories at /path/to/motomedi/datasets/object_detection/task_name/. (if you don't have this directory, please make it.)
 You should make like a below directories.
  /path/to/motomedi/datasets/object_detection/coco/
 After that you should make train.txt on above directory.
 format is

 `/path/to/image_file x1,y1,x2,y2,kind_num`

 if one image have some bounding box then repeat x1~kind_num like below.

 `/path/to/image_file x1,y1,x2,y2,kind_num x1,y1,x2,y2,kind_num`

 train.txt example

  ```
/Volumes/Transcend/coco/train2017/000000020464.jpg 284,304,449,466,17 329,262,398,338,0 428,170,485,256,0 210,233,264,294,0 103,225,133,263,0
/Volumes/Transcend/coco/train2017/000000325583.jpg 51,188,128,335,58 555,245,594,424,56 245,327,348,426,56 228,286,273,319,56 243,312,328,342,56 22,274,245,420,60 368,215,393,229,45 353,208,368,229,75 125,249,157,304,75 1,226,344,412,57 66,288,120,335,75 300,227,421,242,60
/Volumes/Transcend/coco/train2017/000000009738.jpg 47,49,253,421,0 187,65,356,243,37
/Volumes/Transcend/coco/train2017/000000023623.jpg 353,223,506,265,71 3,273,219,381,71
  ...
  ```

### 2. Edit the conf file.<WIP>
 Edit conf file(/path/to/motomedi/training/conf/config.ini)
 Editing points are below.

| No | variable name | example | remark |
|:-----------:|:------------|:------------|:--------|
| 1 | base_dir | /usr/local/wk/motomedi/  | your environment's motomedi path. |
| 2 | image_height | 300 | your images height size. if this doesn't match your image file height, it is ok. Automatically resize on processing using this config. |
| 3 | image_width | 400 | your images width size. and same as image_height. |
| 4 | image_channel_dim | 3 | your image channel dimensions. |
| 5 | batch_size | 12 | cnn's processing batch size. |
| 6 | epoch_num | 10 | cnn's processing epoch number. 1 epoch means using for training all training data. |
| 7 | class_num | 3 | your target classification's result number.(this number correspond with number of directories under t    he datasets/train/ and datasets/test/) |
| 8 | train_path | /usr/local/wk/motomedi/datasets/fruit/train/ | your train data path. |
| 9 | test_path | /usr/local/wk/motomedi/datasets/fruit/test/ | your test data path. |
| 10 | save_dir | /usr/local/wk/motomedi/training/saved/ | your save path. after processing this path save log, model, result, conf file. |

### 3. Do training.
 ```
 export PYTHONPATY=$PYTHONPATY:/path/to/motomedi/training/script
 python train.py
 ```

 please fix from /path/to to your environment's path.

## notes
 This directory is WIP.
 if you want to fix some points then, please make a issue.

## memo
### yolo_v3
### 480*320
batch_size:8 -> ok  
batch_size:16 -> ok  
batch_size:32 -> ng  
batch_size:20 -> ng  

epoch:300, start:06:15:00, end:08:36:00, about 141min  

model_size

```
236M Jan  5 08:16 ep261-loss16.483-val_loss15.922.h5
```

### tiny_yolo
batch_size:32 -> ok  
batch_size:64 -> ok  
batch_size:128 -> ng  

epoch:300, start:09:12:00, end:11:13:00, about 121min

### 960*640
batch_size:8 -> ng  
batch_size:4 -> ok  
batch_size:6 -> ng  
batch_size:5 -> ng  
batch_size:4 & image num:1512 -> ok

epoch:20, start:06:42:00, end:07:22:00, about 40min  
epoch:50, start:07:23:00, end:09:03:00, about 100min  

model_size  

```
236M Jan  1 19:06 trained_weights_final.h5
```

### tiny_yolo
### 960*640
batch_size:4 -> ok  
batch_size:32 -> ng  
batch_size:16 -> ok  

epoch:300, start:04:33:00, end:13:25:00, about 532min

model_size  
```
34M Jan  3 13:23 tiny_trained_weights_final.h5
```
