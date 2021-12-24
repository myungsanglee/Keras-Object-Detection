import os
from glob import glob
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

######################################
# Set GPU
######################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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
# Utility Functions
##################################
def intersection_over_union(boxes1, boxes2):
    """
    Calculation of intersection-over-union

    Arguments:
        boxes1 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [x, y, w, h]
        boxes2 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [x, y, w, h]

    Returns:
        Numpy Array of IoU with shape '(batch, S, S, 1) or (batch, num_boxes, 1) or (num_boxes, 1)'
    """

    box1_xmin = (boxes1[..., 0:1] - boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymin = (boxes1[..., 1:2] - boxes1[..., 3:4]) / 2. # (batch, S, S, 1)
    box1_xmax = (boxes1[..., 0:1] + boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymax = (boxes1[..., 1:2] + boxes1[..., 3:4]) / 2. # (batch, S, S, 1)

    box2_xmin = (boxes2[..., 0:1] - boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymin = (boxes2[..., 1:2] - boxes2[..., 3:4]) / 2. # (batch, S, S, 1)
    box2_xmax = (boxes2[..., 0:1] + boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymax = (boxes2[..., 1:2] + boxes2[..., 3:4]) / 2. # (batch, S, S, 1)

    inter_xmin = np.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
    inter_ymin = np.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
    inter_xmax = np.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
    inter_ymax = np.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)

    inter_area = np.clip((inter_xmax - inter_xmin), 0, 1) * np.clip((inter_ymax - inter_ymin), 0, 1) # (batch, S, S, 1)
    box1_area = np.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = np.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)

def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.4):
    """
    Does Non Max Suppression given boxes

    Arguments:
        boxes (Numpy Array): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, x, y, w, h]
        iou_threshold (float): threshold where predicted boxes is correct
        conf_threshold (float): threshold to remove predicted boxes

    Returns:
        Numpy Array of boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = np.take(boxes, np.where(boxes[..., 1] > conf_threshold)[0], axis=0)

    # sort descending by confidence score
    boxes = np.take(boxes, np.argsort(-boxes[..., 1]), axis=0)

    # get boxes after nms
    boxes_after_nms = np.empty(shape=(0, 6))

    while not(np.less(boxes.shape[0], 1)):
        chosen_box = np.expand_dims(boxes[0], axis=0)
        print(chosen_box[..., 2:].shape)
        tmp_boxes = np.empty(shape=(0, 6))
        for idx in range(1, boxes.shape[0]):
            tmp_box = np.expand_dims(boxes[idx], axis=0)
            if tmp_box[0][0] != chosen_box[0][0] or intersection_over_union(chosen_box[..., 2:], tmp_box[..., 2:]) < iou_threshold:
                tmp_boxes = np.append(tmp_boxes, tmp_box, axis=0)
        boxes = tmp_boxes

        boxes_after_nms = np.append(boxes_after_nms, chosen_box, axis=0)

    return boxes_after_nms


##################################
# Dataset Generator
##################################



##################################
# YOLO v1 Model
##################################
class YoloV1(keras.Model):
    """A subclassed Keras model implementing the YoloV1 architecture

    Attributes:
      num_classes: Number of classes in the dataset
      num_boxes: Number of boxes to predict
      backbone: The backbone to build the YoloV1
    """

    def __init__(self, num_classes, num_boxes, backbone):
        super(YoloV1, self).__init__(name="YoloV1")
        self.backbone = backbone
        
        self.conv_1 = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_1')
        self.conv_2 = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='conv_2')
        self.conv_3 = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_3')
        self.conv_4 = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_4')
        
        self.flatten = keras.layers.Flatten()
        
        self.dense_1 = keras.layers.Dense(units=512, activation='relu', name='dense_1')
        self.dense_2 = keras.layers.Dense(units=1024, activation='relu', name='dense_2')
        self.dropout = keras.layers.Dropout(rate=0.5)
        self.yolo_v1_outputs = keras.layers.Dense(units=7*7*(num_boxes*5 + num_classes), name='yolo_v1_outputs')

    def call(self, images, training=False):
        x = self.backbone(images, training=training)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return self.yolo_v1_outputs(x)


backbone = keras.applications.VGG16(include_top=False, input_shape=(448, 448, 3))
model = YoloV1(20, 2, backbone)
tmp_inputs = keras.Input(shape=(448, 448, 3))
model(tmp_inputs)
model.summary()
model.save('test')

##################################
# Loss Function
##################################
class YoloV1Loss(keras.losses.Loss):
    def __init__(self, c=20):
        super(YoloV1Loss, self).__init__(name="YoloV1Loss")
        self.C = c

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

        self.bbox1_start_index = c + 1
        self.bbox1_confidence_index = c
        self.bbox2_start_index = c + 1 + 5
        self.bbox2_confidence_index = c + 5

    def call(self, y_true, y_pred):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        y_pred = tf.reshape(y_pred, [-1, 7, 7, 30])

        # # Add Activation Functions
        # # Softmax for class
        # class_pred = y_pred[..., :self.C]
        # class_pred = tf.keras.activations.softmax(class_pred)
        #
        # # Sigmoid for confidence and bbox
        # conf_bbox_pred = y_pred[..., self.C:]
        # conf_bbox_pred = tf.keras.activations.sigmoid(conf_bbox_pred)
        #
        # # Concat
        # y_pred = tf.concat([class_pred, conf_bbox_pred], axis=-1)

        # Calculate IoU for the two pred bbox with true bbox
        iou_b1 = intersection_over_union(y_true[..., self.bbox1_start_index:self.bbox1_start_index + 4],
                                         y_pred[..., self.bbox1_start_index:self.bbox1_start_index + 4])  # (batch, S, S, 1)
        iou_b2 = intersection_over_union(y_true[..., self.bbox1_start_index:self.bbox1_start_index + 4],
                                         y_pred[..., self.bbox2_start_index:self.bbox2_start_index + 4])  # (batch, S, S, 1)
        ious = tf.concat([iou_b1, iou_b2], axis=-1)  # (batch, S, S, 2)

        # Get best box index
        best_box_index = tf.math.argmax(ious, axis=-1)  # (batch, S, S)
        best_box_index = tf.expand_dims(best_box_index, axis=-1)  # (batch, S, S, 1)
        best_box_index = tf.cast(best_box_index, dtype=tf.float32)

        # Get true box confidence(object exist in cell)
        # in paper this is Iobj_i
        y_true_conf = y_true[..., self.bbox1_confidence_index:self.bbox1_confidence_index + 1]  # (batch, S, S, 1)

        # ============================ #
        #   FOR BOX COORDINATES Loss   #
        # ============================ #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_pred = ((1. - best_box_index) * y_pred[..., self.bbox1_start_index:self.bbox1_start_index + 4] +
                    best_box_index * y_pred[..., self.bbox2_start_index:self.bbox2_start_index + 4])  # (batch, S, S, 4)

        box_true = y_true[..., self.bbox1_start_index:self.bbox1_start_index + 4]  # (batch, S, S, 4)

        xy_loss = y_true_conf * tf.math.square(box_true[..., 0:2] - box_pred[..., 0:2])  # (batch, S, S, 2)
        xy_loss = tf.math.reduce_sum(xy_loss)  # scalar value

        # Take sqrt of width, height of boxes to ensure that
        wh_loss = y_true_conf * tf.math.square(tf.math.sqrt(box_true[..., 2:4]) -
                                               (tf.math.sign(box_pred[..., 2:4]) * tf.math.sqrt(
                                                   tf.math.abs(box_pred[..., 2:4]) + 1e-6)))  # (batch, S, S, 2)
        wh_loss = tf.math.reduce_sum(wh_loss)  # scalar value

        box_loss = xy_loss + wh_loss  # scalar value

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # conf_pred is the confidence score for the bbox with highest IoU
        conf_pred = ((1. - best_box_index) * y_pred[..., self.bbox1_confidence_index:self.bbox1_confidence_index + 1] +
                     best_box_index * y_pred[..., self.bbox2_confidence_index:self.bbox2_confidence_index + 1])  # (batch, S, S, 1)

        object_loss = y_true_conf * tf.math.square(1 - conf_pred)  # (batch, S, S, 1)
        object_loss = tf.math.reduce_sum(object_loss)  # scalar value

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = (1 - y_true_conf) * tf.math.square(
            0 - y_pred[..., self.bbox1_confidence_index:self.bbox1_confidence_index + 1])  # (batch, S, S, 1)
        no_object_loss += (1 - y_true_conf) * tf.math.square(
            0 - y_pred[..., self.bbox2_confidence_index:self.bbox2_confidence_index + 1])  # (batch, S, S, 1)
        no_object_loss = tf.math.reduce_sum(no_object_loss)  # scalar value

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = y_true_conf * tf.math.square(y_true[..., :self.C] - y_pred[..., :self.C])  # (batch, S, S, C)
        class_loss = tf.math.reduce_sum(class_loss)  # scalar value

        loss = (self.lambda_coord * box_loss) + \
               object_loss + \
               (self.lambda_noobj * no_object_loss) + \
               class_loss

        return loss
    
##################################
# Train
##################################
# yolo_loss = YoloV1Loss()
# optimizer = keras.optimizers.Adam(learning_rate=0.001)
# model.compile(loss=yolo_loss, optimizer=optimizer)

# model.summary()