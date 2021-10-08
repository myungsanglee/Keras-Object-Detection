from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from glob import glob
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from utils import non_max_suppression, get_all_bboxes, non_max_suppression_2, get_all_bboxes_numpy, non_max_suppression_numpy
from dataset import YoloV1Generator2
from model import yolov1, mobilenet_v2_yolo_v1, test_model

######################################
# Set GPU
######################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


######################################
# Set GPU Memory
######################################
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
#   except RuntimeError as e:
#     print(e)


##################################
# Variables
##################################
"""
S is split size of image (in paper 7),
B is number of boxes (in paper 2),
C is number of classes (in paper and VOC dataset is 20),
"""
S = 7
B = 2
C = 20

input_shape = (448, 448, 3)
output_shape = (S, S, C + (B * 5))

batch_size = 1

# path variables
cur_dir = os.getcwd()
save_model_dir = os.path.join(cur_dir, "saved_models/2021-10-01 18:16:10")
train_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/train"
val_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/val"
test_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/test"
obj_name_path = "/home/fssv2/myungsang/datasets/voc_2007/voc.names"


##################################
# Get Dataset Generator
##################################
jpg_data_list = glob(train_dir + "/*.jpg")

test_generator = YoloV1Generator2(train_dir,
                                  input_shape=input_shape,
                                  batch_size=batch_size,
                                  drop_remainder=False,
                                  augment=False,
                                  shuffle=False)

with open(obj_name_path, 'r') as f:
    obj_name_list = f.readlines()
obj_name_list = [data.strip() for data in obj_name_list]
print(obj_name_list)


##################################
# Get model
##################################
# Get trained model
# model_dir_list = glob(save_model_dir + "/*")
# model_dir = model_dir_list[0]
model_list = glob(save_model_dir + "/*")
model_list = sorted(model_list)
best_model = model_list[-1]
print("Best Model Name: {}".format(best_model))
# model = keras.models.load_model(best_model, compile=False)
# model = yolov1(input_shape, output_shape)
model = mobilenet_v2_yolo_v1(input_shape, output_shape)
# model = test_model(input_shape, output_shape)
model.load_weights(best_model)


##################################
# Get bbox img function
##################################
def get_bbox_img(img_path, bboxes, class_name_list):
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    for bbox in bboxes:
        class_name = class_name_list[int(bbox[0])]
        confidence_score = bbox[1]
        x = bbox[2]
        y = bbox[3]
        w = bbox[4]
        h = bbox[5]
        xmin = int((x - (w / 2)) * width)
        ymin = int((y - (h / 2)) * height)
        xmax = int((x + (w / 2)) * width)
        ymax = int((y + (h / 2)) * height)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255))
        img = cv2.putText(img, "{0}, {1:.2f}".format(class_name, confidence_score), (xmin, ymin + 15),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=(0, 0, 255))
    return img


##################################
# Inference
##################################
for idx in range(test_generator.__len__()):

    # Get Sample Dataset
    sample_x_true, sample_y_true = test_generator.__getitem__(idx)

    # inference
    start_time = time.time()
    predictions = model(sample_x_true, training=False)
    print("Inference FPS: {:.1f}".format(1 / (time.time() - start_time)))

    predictions_numpy = predictions.numpy()

    # get bboxes
    second_time = time.time()
    predictions = tf.reshape(predictions, [-1, 7, 7, 30])
    pred_bboxes = get_all_bboxes(predictions)
    pred_bboxes = non_max_suppression_2(pred_bboxes[0], threshold=0.4, iou_threshold=0.5)
    print("NMS FPS: {:.1f}".format(1 / (time.time() - second_time)))

    # get bboxes numpy
    second_time = time.time()
    predictions_numpy = np.reshape(predictions_numpy, (-1, 7, 7, 30))
    pred_bboxes_numpy = get_all_bboxes_numpy(predictions_numpy)
    pred_bboxes_numpy = non_max_suppression_numpy(pred_bboxes_numpy[0], threshold=0.4, iou_threshold=0.5)
    print("NMS Numpy FPS: {:.1f}".format(1 / (time.time() - second_time)))

    # Get bbox img
    x_true_img = get_bbox_img(jpg_data_list[idx], pred_bboxes, obj_name_list)
    print("FPS: {:.1f}".format(1/(time.time() - start_time)))

    # Show
    cv2.imshow("Result", x_true_img)
    key = cv2.waitKey(0)
    if key == 27:
        break
