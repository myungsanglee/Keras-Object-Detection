from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import intersection_over_union
import numpy as np


##################################
# YOLO v1 Loss
##################################
@tf.function
def yolo_loss(y_true, y_pred, C=20):
    # These are from Yolo paper, signifying how much we should
    # pay loss for no object (noobj) and the box coordinates (coord)
    lambda_noobj = 0.5
    lambda_coord = 5

    bbox1_start_index = C + 1
    bbox1_confidence_index = C
    bbox2_start_index = C + 1 + 5
    bbox2_confidence_index = C + 5

    # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
    y_pred = tf.reshape(y_pred, [-1, 7, 7, 30])

    # Calculate IoU for the two pred bbox with true bbox
    iou_b1 = intersection_over_union(y_true[..., bbox1_start_index:bbox1_start_index+4],
                                     y_pred[..., bbox1_start_index:bbox1_start_index+4]) # (batch, S, S, 1)
    iou_b2 = intersection_over_union(y_true[..., bbox1_start_index:bbox1_start_index+4],
                                     y_pred[..., bbox2_start_index:bbox2_start_index+4]) # (batch, S, S, 1)
    ious = tf.concat([iou_b1, iou_b2], axis=-1) # (batch, S, S, 2)

    # Get best box index
    best_box_index = tf.math.argmax(ious, axis=-1) # (batch, S, S)
    best_box_index = tf.expand_dims(best_box_index, axis=-1) # (batch, S, S, 1)
    best_box_index = tf.cast(best_box_index, dtype=tf.float32)

    # Get true box confidence(object exist in cell)
    # in paper this is Iobj_i
    y_true_conf = y_true[..., bbox1_confidence_index:bbox1_confidence_index+1] # (batch, S, S, 1)

    # ============================ #
    #   FOR BOX COORDINATES Loss   #
    # ============================ #

    # Set boxes with no object in them to 0. We only take out one of the two
    # predictions, which is the one with highest Iou calculated previously.
    box_pred = ((1. - best_box_index) * y_pred[..., bbox1_start_index:bbox1_start_index+4] +
                best_box_index * y_pred[..., bbox2_start_index:bbox2_start_index+4]) # (batch, S, S, 4)

    box_true = y_true[..., bbox1_start_index:bbox1_start_index+4] # (batch, S, S, 4)

    xy_loss = y_true_conf * tf.math.square(box_true[..., 0:2] - box_pred[..., 0:2]) # (batch, S, S, 2)
    xy_loss = tf.math.reduce_sum(xy_loss) # scalar value

    # Take sqrt of width, height of boxes to ensure that
    wh_loss = y_true_conf * tf.math.square(tf.math.sqrt(box_true[..., 2:4]) -
                                           (tf.math.sign(box_pred[..., 2:4]) * tf.math.sqrt(tf.math.abs(box_pred[..., 2:4]) + 1e-6))) # (batch, S, S, 2)
    wh_loss = tf.math.reduce_sum(wh_loss) # scalar value

    box_loss = xy_loss + wh_loss  # scalar value

    # ==================== #
    #   FOR OBJECT LOSS    #
    # ==================== #

    # conf_pred is the confidence score for the bbox with highest IoU
    conf_pred = ((1. - best_box_index) * y_pred[..., bbox1_confidence_index:bbox1_confidence_index+1] +
                 best_box_index * y_pred[..., bbox2_confidence_index:bbox2_confidence_index+1]) # (batch, S, S, 1)

    object_loss = y_true_conf * tf.math.square(1 - conf_pred) # (batch, S, S, 1)
    object_loss = tf.math.reduce_sum(object_loss) # scalar value

    # ======================= #
    #   FOR NO OBJECT LOSS    #
    # ======================= #

    no_object_loss = (1 - y_true_conf) * tf.math.square(0 - y_pred[..., bbox1_confidence_index:bbox1_confidence_index+1]) # (batch, S, S, 1)
    no_object_loss += (1 - y_true_conf) * tf.math.square(0 - y_pred[..., bbox2_confidence_index:bbox2_confidence_index+1]) # (batch, S, S, 1)
    no_object_loss = tf.math.reduce_sum(no_object_loss) # scalar value

    # ================== #
    #   FOR CLASS LOSS   #
    # ================== #

    class_loss = y_true_conf * tf.math.square(y_true[..., :C] - y_pred[..., :C]) # (batch, S, S, C)
    class_loss = tf.math.reduce_sum(class_loss) # scalar value

    loss = (lambda_coord * box_loss) + \
           object_loss + \
           (lambda_noobj * no_object_loss) + \
           class_loss

    return loss


class YoloV1Loss(tf.keras.losses.Loss):
    def __init__(self, c=20, name="yolo_v1_loss"):
        super(YoloV1Loss, self).__init__(name=name)
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


if __name__ == "__main__":
    y_true = np.zeros((1, 7, 7, 13))
    y_true[:, 0, 0, 2] = 1 # class
    y_true[:, 0, 0, 3] = 1 # confidence
    y_true[:, 0, 0, 4:8] = (0.5, 0.5, 0.1, 0.1)
    print("y_true:\n{}".format(y_true))

    y_pred = np.zeros((1, 7, 7, 13))
    y_pred[:, 0, 0, 2] = 0.6  # class
    y_pred[:, 0, 0, 3] = 0.7  # confidence
    y_pred[:, 0, 0, 4:8] = (0.49, 0.49, 0.09, 0.09)
    y_pred[:, 0, 0, 9] = 0.4  # confidence
    y_pred[:, 0, 0, 9:13] = (0.45, 0.45, 0.09, 0.09)
    print("y_pred:\n{}".format(y_pred))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    print(yolo_loss(y_true, y_pred, C=3))

    loss = YoloV1Loss(c=3)
    print(loss(y_true, y_pred))
