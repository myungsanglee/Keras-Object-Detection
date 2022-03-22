from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import intersection_over_union
import numpy as np
from tensorflow import keras


##################################
# YOLO v1 Loss
##################################
# @tf.function
# def yolo_loss(y_true, y_pred, C=20):
#     # These are from Yolo paper, signifying how much we should
#     # pay loss for no object (noobj) and the box coordinates (coord)
#     lambda_noobj = 0.5
#     lambda_coord = 5

#     bbox1_start_index = C + 1
#     bbox1_confidence_index = C
#     bbox2_start_index = C + 1 + 5
#     bbox2_confidence_index = C + 5

#     # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
#     # y_pred = tf.reshape(y_pred, [-1, 7, 7, 30])

#     # Calculate IoU for the two pred bbox with true bbox
#     iou_b1 = intersection_over_union(y_true[..., bbox1_start_index:bbox1_start_index+4],
#                                      y_pred[..., bbox1_start_index:bbox1_start_index+4]) # (batch, S, S, 1)
#     iou_b2 = intersection_over_union(y_true[..., bbox1_start_index:bbox1_start_index+4],
#                                      y_pred[..., bbox2_start_index:bbox2_start_index+4]) # (batch, S, S, 1)
#     ious = tf.concat([iou_b1, iou_b2], axis=-1) # (batch, S, S, 2)

#     # Get best box index
#     best_box_index = tf.math.argmax(ious, axis=-1) # (batch, S, S)
#     best_box_index = tf.expand_dims(best_box_index, axis=-1) # (batch, S, S, 1)
#     best_box_index = tf.cast(best_box_index, dtype=tf.float32)

#     # Get true box confidence(object exist in cell)
#     # in paper this is Iobj_i
#     y_true_conf = y_true[..., bbox1_confidence_index:bbox1_confidence_index+1] # (batch, S, S, 1)

#     # ============================ #
#     #   FOR BOX COORDINATES Loss   #
#     # ============================ #

#     # Set boxes with no object in them to 0. We only take out one of the two
#     # predictions, which is the one with highest Iou calculated previously.
#     box_pred = ((1. - best_box_index) * y_pred[..., bbox1_start_index:bbox1_start_index+4] +
#                 best_box_index * y_pred[..., bbox2_start_index:bbox2_start_index+4]) # (batch, S, S, 4)

#     box_true = y_true[..., bbox1_start_index:bbox1_start_index+4] # (batch, S, S, 4)

#     xy_loss = y_true_conf * tf.math.square(box_true[..., 0:2] - box_pred[..., 0:2]) # (batch, S, S, 2)
#     xy_loss = tf.math.reduce_sum(xy_loss) # scalar value

#     # Take sqrt of width, height of boxes to ensure that
#     wh_loss = y_true_conf * tf.math.square(tf.math.sqrt(box_true[..., 2:4]) -
#                                            (tf.math.sign(box_pred[..., 2:4]) * tf.math.sqrt(tf.math.abs(box_pred[..., 2:4]) + 1e-6))) # (batch, S, S, 2)
#     wh_loss = tf.math.reduce_sum(wh_loss) # scalar value

#     box_loss = xy_loss + wh_loss  # scalar value

#     # ==================== #
#     #   FOR OBJECT LOSS    #
#     # ==================== #

#     # conf_pred is the confidence score for the bbox with highest IoU
#     conf_pred = ((1. - best_box_index) * y_pred[..., bbox1_confidence_index:bbox1_confidence_index+1] +
#                  best_box_index * y_pred[..., bbox2_confidence_index:bbox2_confidence_index+1]) # (batch, S, S, 1)

#     object_loss = y_true_conf * tf.math.square(1 - conf_pred) # (batch, S, S, 1)
#     object_loss = tf.math.reduce_sum(object_loss) # scalar value

#     # ======================= #
#     #   FOR NO OBJECT LOSS    #
#     # ======================= #

#     no_object_loss = (1 - y_true_conf) * tf.math.square(0 - y_pred[..., bbox1_confidence_index:bbox1_confidence_index+1]) # (batch, S, S, 1)
#     no_object_loss += (1 - y_true_conf) * tf.math.square(0 - y_pred[..., bbox2_confidence_index:bbox2_confidence_index+1]) # (batch, S, S, 1)
#     no_object_loss = tf.math.reduce_sum(no_object_loss) # scalar value

#     # ================== #
#     #   FOR CLASS LOSS   #
#     # ================== #

#     class_loss = y_true_conf * tf.math.square(y_true[..., :C] - y_pred[..., :C]) # (batch, S, S, C)
#     class_loss = tf.math.reduce_sum(class_loss) # scalar value

#     loss = (lambda_coord * box_loss) + \
#            object_loss + \
#            (lambda_noobj * no_object_loss) + \
#            class_loss

#     return loss


class YoloV1Loss(keras.losses.Loss):
    """YoloV1 Loss Function

    Arguments:
      num_classes: Number of classes in the dataset
      num_boxes: Number of boxes to predict
    """

    def __init__(self, num_classes=20, num_boxes=2):
        super(YoloV1Loss, self).__init__(name="YoloV1Loss")
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
        self.batch_size = 0

    def call(self, y_true, y_pred):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5)) when inputted
        # y_pred = tf.reshape(y_pred, [-1, 7, 7, 30])
        self.batch_size = y_true.shape[0]

        # Calculate IoU for the prediction boxes with true boxes
        ious = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for idx in tf.range(self.num_boxes):
            iou = intersection_over_union(
                y_true[..., self.num_classes+1:self.num_classes+5],
                y_pred[..., self.num_classes+(1+(5*idx)):self.num_classes+(1+(5*idx))+4]
            )
            ious = ious.write(ious.size(), iou)
        ious = ious.stack() # (num_boxes, batch, S, S, 1)

        # Get best iou index & one_hot
        best_iou_idx = tf.math.argmax(ious, axis=0) # (batch, S, S, 1)
        best_iou_one_hot = tf.reshape(tf.one_hot(best_iou_idx, self.num_boxes), shape=(-1, 7, 7, self.num_boxes)) # (batch, S, S, num_boxes)
        
        # Get prediction box & iou
        # pred_box = tf.zeros(shape=[self.batch_size, 7, 7, 4])
        # pred_conf = tf.zeros(shape=[self.batch_size, 7, 7, 1])
        # pred_iou = tf.zeros(shape=[self.batch_size, 7, 7, 1])
        pred_box = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        pred_conf = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        pred_iou = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for idx in tf.range(self.num_boxes):
            # pred_box += best_iou_one_hot[..., idx:idx+1] * y_pred[..., self.num_classes+(1+(5*idx)):self.num_classes+(1+(5*idx))+4]
            # pred_conf += best_iou_one_hot[..., idx:idx+1] * y_pred[..., self.num_classes+(5*idx):self.num_classes+(5*idx)+1]
            # pred_iou += best_iou_one_hot[..., idx:idx+1] * ious[idx]
            pred_box = pred_box.write(pred_box.size(), best_iou_one_hot[..., idx:idx+1] * y_pred[..., self.num_classes+(1+(5*idx)):self.num_classes+(1+(5*idx))+4])
            pred_conf = pred_conf.write(pred_conf.size(), best_iou_one_hot[..., idx:idx+1] * y_pred[..., self.num_classes+(5*idx):self.num_classes+(5*idx)+1])
            pred_iou = pred_iou.write(pred_iou.size(), best_iou_one_hot[..., idx:idx+1] * ious[idx])
        pred_box = tf.math.reduce_sum(pred_box.stack(), axis=0)
        pred_conf = tf.math.reduce_sum(pred_conf.stack(), axis=0)
        pred_iou = tf.math.reduce_sum(pred_iou.stack(), axis=0)

        # Get true box
        true_box = y_true[..., self.num_classes+1:self.num_classes+5] # (batch, S, S, 4)

        # Get true box confidence(object exist in cell)
        # in paper this is 1obj_ij
        obj = y_true[..., self.num_classes:self.num_classes + 1]  # (batch, S, S, 1)
        noobj = 1 - obj  # (batch, S, S, 1)

        # ============================ #
        #   FOR BOX COORDINATES Loss   #
        # ============================ #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        xy_loss = obj * tf.math.square(true_box[..., 0:2] - pred_box[..., 0:2])  # (batch, S, S, 2)
        xy_loss = tf.math.reduce_sum(xy_loss)  # scalar value
        # print(f"xy_loss: {xy_loss}")

        # Take sqrt of width, height of boxes to ensure that
        wh_loss = obj * tf.math.square(
            tf.math.sqrt(true_box[..., 2:4]) - (tf.math.sign(pred_box[..., 2:4]) * tf.math.sqrt(tf.math.abs(pred_box[..., 2:4]) + 1e-6))
        )  # (batch, S, S, 2)
        wh_loss = tf.math.reduce_sum(wh_loss)  # scalar value
        # print(f"wh_loss: {wh_loss}")

        box_loss = xy_loss + wh_loss  # scalar value

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # conf_pred is the confidence score for the bbox with highest IoU
        object_loss = obj * tf.math.square(pred_iou - pred_conf)  # (batch, S, S, 1)
        object_loss = tf.math.reduce_sum(object_loss)  # scalar value
        # print(f"object_loss: {object_loss}")

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = noobj * tf.math.square(0 - pred_conf)  # (batch, S, S, 1)
        no_object_loss = tf.math.reduce_sum(no_object_loss)  # scalar value
        # print(f"no_object_loss: {no_object_loss}")


        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = obj * tf.math.square(y_true[..., :self.num_classes] - y_pred[..., :self.num_classes])  # (batch, S, S, C)
        class_loss = tf.math.reduce_sum(class_loss)  # scalar value
        # print(f"class_loss: {class_loss}")

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

    loss = YoloV1Loss(num_classes=3, num_boxes=2)
    print(loss(y_true, y_pred))
