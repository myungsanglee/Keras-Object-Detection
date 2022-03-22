import time
from collections import Counter

import numpy as np
import tensorflow as tf
import cv2


@tf.function
def intersection_over_union(boxes1, boxes2):
    """Calculation of intersection-over-union

    Arguments:
        boxes1 (Tensor): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]
        boxes2 (Tensor): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]

    Returns:
        Tensor: IoU with shape '(batch, S, S, 1) or (batch, num_boxes, 1) or (num_boxes, 1)'
    """
    if not tf.is_tensor(boxes1) or not tf.is_tensor(boxes2):
        boxes1 = tf.cast(boxes1, dtype=tf.float32)
        boxes2 = tf.cast(boxes2, dtype=tf.float32)

    box1_xmin = (boxes1[..., 0:1] - boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymin = (boxes1[..., 1:2] - boxes1[..., 3:4]) / 2. # (batch, S, S, 1)
    box1_xmax = (boxes1[..., 0:1] + boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymax = (boxes1[..., 1:2] + boxes1[..., 3:4]) / 2. # (batch, S, S, 1)

    box2_xmin = (boxes2[..., 0:1] - boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymin = (boxes2[..., 1:2] - boxes2[..., 3:4]) / 2. # (batch, S, S, 1)
    box2_xmax = (boxes2[..., 0:1] + boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymax = (boxes2[..., 1:2] + boxes2[..., 3:4]) / 2. # (batch, S, S, 1)

    inter_xmin = tf.math.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
    inter_ymin = tf.math.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
    inter_xmax = tf.math.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
    inter_ymax = tf.math.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)

    inter_area = tf.clip_by_value((inter_xmax - inter_xmin), 0, 1) * tf.clip_by_value((inter_ymax - inter_ymin), 0, 1) # (batch, S, S, 1)
    box1_area = tf.math.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = tf.math.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)


def intersection_over_union_numpy(boxes1, boxes2):
    """Calculation of intersection-over-union

    Arguments:
        boxes1 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]
        boxes2 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]

    Returns:
        Numpy Array: IoU with shape '(batch, S, S, 1) or (batch, num_boxes, 1) or (num_boxes, 1)'
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


@tf.function
def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.4):
    """Does Non Max Suppression given bboxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
        iou_threshold (float): threshold where predicted boxes is correct
        conf_threshold (float): threshold to remove predicted boxes

    Returns:
        Tensor: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """
    if not tf.is_tensor(boxes):
        boxes = tf.cast(boxes, dtype=tf.float32)

    # boxes smaller than the threshold are removed
    boxes = tf.gather(boxes, tf.reshape(tf.where(boxes[..., 1] > conf_threshold), shape=(-1,)))

    # sort descending by confidence score
    boxes = tf.gather(boxes, tf.argsort(boxes[..., 1], direction='DESCENDING'))

    # get boxes after nms
    boxes_after_nms = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    while not(tf.less(tf.shape(boxes)[0], 1)):
        chosen_box = boxes[0]
        tmp_boxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for idx in tf.range(1, tf.shape(boxes)[0]):
            bbox = boxes[idx]
            if bbox[0] != chosen_box[0] or intersection_over_union(chosen_box[2:], bbox[2:]) < iou_threshold:
                tmp_boxes = tmp_boxes.write(tmp_boxes.size(), bbox)
        boxes = tmp_boxes.stack()

        boxes_after_nms = boxes_after_nms.write(boxes_after_nms.size(), chosen_box)

    return boxes_after_nms.stack()


def non_max_suppression_numpy(boxes, iou_threshold=0.5, conf_threshold=0.4):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Numpy Array): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
        iou_threshold (float): threshold where predicted boxes is correct
        conf_threshold (float): threshold to remove predicted boxes

    Returns:
        Numpy Array: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = np.take(boxes, np.where(boxes[..., 1] > conf_threshold)[0], axis=0)

    # sort descending by confidence score
    boxes = np.take(boxes, np.argsort(-boxes[..., 1]), axis=0)

    # get boxes after nms
    boxes_after_nms = np.empty(shape=(0, 6))

    while not(np.less(boxes.shape[0], 1)):
        chosen_box = np.expand_dims(boxes[0], axis=0)
        tmp_boxes = np.empty(shape=(0, 6))
        for idx in range(1, boxes.shape[0]):
            tmp_box = np.expand_dims(boxes[idx], axis=0)
            if tmp_box[0][0] != chosen_box[0][0] or intersection_over_union(chosen_box[..., 2:], tmp_box[..., 2:]) < iou_threshold:
                tmp_boxes = np.append(tmp_boxes, tmp_box, axis=0)
        boxes = tmp_boxes

        boxes_after_nms = np.append(boxes_after_nms, chosen_box, axis=0)

    return boxes_after_nms


@tf.function
def decode_predictions(predictions, num_classes, num_boxes=2):
    """decodes predictions of the YOLO v1 model
    
    Converts bounding boxes output from Yolo with
    an image split size of GRID into entire image ratios
    rather than relative to cell ratios.

    Arguments:
        predictions (Tensor): predictions of the YOLO v1 model with shape  '(1, 7, 7, (num_boxes*5 + num_classes))'
        num_classes: Number of classes in the dataset
        num_boxes: Number of boxes to predict

    Returns:
        Tensor: boxes after decoding predictions each grid cell with shape '(batch, S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
    """
    
    if not tf.is_tensor(predictions):
        predictions = tf.cast(predictions, dtype=tf.float32)

    # Get class indexes
    class_indexes = tf.math.argmax(predictions[..., :num_classes], axis=-1) # (batch, S, S)
    class_indexes = tf.expand_dims(class_indexes, axis=-1) # (batch, S, S, 1)
    class_indexes = tf.cast(class_indexes, dtype=np.float32)
    
    # Get best confidence one-hot
    confidences = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for idx in tf.range(num_boxes):
        confidence = predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1]
        confidences = confidences.write(confidences.size(), confidence)
    confidences = confidences.stack() # (num_boxes, batch, S, S, 1)
    best_conf_idx = tf.math.argmax(confidences, axis=0) # (batch, S, S, 1)
    best_conf_one_hot = tf.reshape(tf.one_hot(best_conf_idx, num_boxes), shape=(-1, 7, 7, num_boxes)) # (batch, S, S, num_boxes)
    
    # Get prediction box & confidence
    # pred_box = tf.zeros(shape=[1, 7, 7, 4])
    # pred_conf = tf.zeros(shape=[1, 7, 7, 1])
    pred_box = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    pred_conf = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for idx in tf.range(num_boxes):
        # pred_box += best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(1+(5*idx)):num_classes+(1+(5*idx))+4]
        # pred_conf += best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1]
        pred_box = pred_box.write(pred_box.size(), best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(1+(5*idx)):num_classes+(1+(5*idx))+4])
        pred_conf = pred_conf.write(pred_conf.size(), best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1])
    pred_box = tf.math.reduce_sum(pred_box.stack(), axis=0) # (1, 7, 7, 4)
    pred_conf = tf.math.reduce_sum(pred_conf.stack(), axis=0) # (1, 7, 7, 1)

    # Get cell indexes array
    base_arr = tf.map_fn(fn=lambda x: tf.range(x, x + 7), elems=tf.zeros(7))
    x_cell_indexes = tf.reshape(base_arr, shape=(7, 7, 1)) # (S, S, 1)

    y_cell_indexes = tf.transpose(base_arr)
    y_cell_indexes = tf.reshape(y_cell_indexes, shape=(7, 7, 1)) # (S, S, 1)

    # Convert x, y ratios to YOLO ratios
    x = 1 / 7 * (pred_box[..., :1] + x_cell_indexes) # (batch, S, S, 1)
    y = 1 / 7 * (pred_box[..., 1:2] + y_cell_indexes) # (batch, S, S, 1)
    
    pred_box = tf.concat([x, y, pred_box[..., 2:4]], axis=-1) # (batch, S, S, 4)
    
    # Concatenate result
    pred_result = tf.concat([class_indexes, pred_conf, pred_box], axis=-1) # (batch, S, S, 6)

    # Get all bboxes
    pred_result = tf.reshape(pred_result, shape=(-1, 7*7, 6)) # (batch, S*S, 6)
    
    return pred_result


def decode_predictions_numpy(predictions, num_classes, num_boxes=2):
    """decodes predictions of the YOLO v1 model
    
    Converts bounding boxes output from Yolo with
    an image split size of GRID into entire image ratios
    rather than relative to cell ratios.

    Arguments:
        predictions (Tensor): predictions of the YOLO v1 model with shape  '(1, 7, 7, (num_boxes*5 + num_classes))'
        num_classes: Number of classes in the dataset
        num_boxes: Number of boxes to predict

    Returns:
        Tensor: boxes after decoding predictions each grid cell with shape '(batch, S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
    """

    # Get class indexes
    class_indexes = np.argmax(predictions[..., :num_classes], axis=-1) # (batch, S, S)
    class_indexes = np.expand_dims(class_indexes, axis=-1) # (batch, S, S, 1)
    class_indexes = class_indexes.astype(np.float32)
    
    # Get best confidence one-hot
    confidences = []
    for idx in np.arange(num_boxes):
        confidence = predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1]
        confidences.append(confidence)
    confidences = np.array(confidences, np.float32) # (num_boxes, batch, S, S, 1)
    best_conf_idx = np.argmax(confidences, axis=0) # (batch, S, S, 1)
    best_conf_one_hot = np.reshape(np.eye(num_boxes)[best_conf_idx.reshape(-1).astype(np.int)], (best_conf_idx.shape[0], best_conf_idx.shape[1], best_conf_idx.shape[2], num_boxes)) # (batch, S, S, num_boxes)
    
    # Get prediction box & confidence
    pred_box = np.zeros(shape=[1, 7, 7, 4])
    pred_conf = np.zeros(shape=[1, 7, 7, 1])
    for idx in np.arange(num_boxes):
        pred_box += best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(1+(5*idx)):num_classes+(1+(5*idx))+4]
        pred_conf += best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1]

    # Get cell indexes array
    base_arr = np.arange(7).reshape((1, -1)).repeat(7, axis=0)
    x_cell_indexes = np.expand_dims(base_arr, axis=-1)  # (S, S, 1)

    y_cell_indexes = np.transpose(base_arr)
    y_cell_indexes = np.expand_dims(y_cell_indexes, axis=-1)  # (S, S, 1)

    # Convert x, y ratios to YOLO ratios
    x = 1 / 7 * (pred_box[..., :1] + x_cell_indexes) # (batch, S, S, 1)
    y = 1 / 7 * (pred_box[..., 1:2] + y_cell_indexes) # (batch, S, S, 1)

    pred_box = np.concatenate([x, y, pred_box[..., 2:4]], axis=-1) # (batch, S, S, 4)
    
    # Concatenate result
    pred_result = np.concatenate([class_indexes, pred_conf, pred_box], axis=-1) # (batch, S, S, 6)

    # Get all bboxes
    pred_result = np.reshape(pred_result, (-1, 7*7, 6)) # (batch, S*S, 6)
    
    return pred_result


@tf.function
def change_tensor(tensor_1d, idx_col):
    """change the value of a specific column in a tensor to 1

    Arguments:
        tensor_1d (Tensor): 1D Tensor to change
        idx_col (Tensor): index of specific column to change

    Returns:
        Tensor: changed tensor_1d
    """
    dst = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    idx = tf.constant(0)
    for value in tensor_1d:
        if idx == idx_col:
            dst = dst.write(dst.size(), tf.constant(1))
        else:
            dst = dst.write(dst.size(), value)
        idx += 1
    return dst.stack()


# @tf.function(input_signature=[tf.TensorSpec(shape=(None, 7)), tf.TensorSpec(shape=(None, 7))])
@tf.function
def mean_average_precision(true_boxes, pred_boxes, num_classes, iou_threshold=0.5):
    """Calculates mean average precision

    Arguments:
        true_boxes (Tensor): Tensor of all boxes with all images (None, 7), specified as [img_idx, class_idx, confidence_score, cx, cy, w, h]
        pred_boxes (Tensor): Similar as true_bboxes
        num_classes (int): number of classes
        iou_threshold (float): threshold where predicted boxes is correct

    Returns:
        Float: mAP value across all classes given a specific IoU threshold
    """
    
    assert tf.is_tensor(true_boxes) and tf.is_tensor(pred_boxes)

    # list storing all AP for respective classes
    average_precisions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    # used for numerical stability later on
    epsilon = 1e-6

    for c in tf.range(num_classes, dtype=tf.float32):
        tf.print('Calculating AP: ', c, ' / ', num_classes)
        
        # detections, ground_truths variables in specific class
        detections = tf.gather(pred_boxes, tf.reshape(tf.where(pred_boxes[..., 1] == c), shape=(-1,)))
        ground_truths = tf.gather(true_boxes, tf.reshape(tf.where(true_boxes[..., 1] == c), shape=(-1,)))

        # If none exists for this class then we can safely skip
        total_true_boxes = tf.cast(tf.shape(ground_truths)[0], dtype=tf.float32)
        if total_true_boxes == 0.:
            average_precisions = average_precisions.write(average_precisions.size(), tf.constant(0, dtype=tf.float32))
            continue

        # tf.print(c, ' class ground truths size: ', tf.shape(ground_truths)[0])
        # tf.print(c, ' class detections size: ', tf.shape(detections)[0])

        # Get the number of true boxes by image index
        img_idx, idx, count = tf.unique_with_counts(ground_truths[..., 0])
        img_idx = tf.cast(img_idx, dtype=tf.int32)

        # Convert idx to idx tensor for find num of true boxes by img idx
        idx_tensor = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        for i in tf.range(tf.math.reduce_max(idx) + 1):
            idx_tensor = idx_tensor.write(idx_tensor.size(), i)
        idx_tensor = idx_tensor.stack()

        # Get hash table: key - img_idx, value - idx_tensor
        table = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.int32,
            value_dtype=tf.int32,
            default_value=-1,
            empty_key=-1,
            deleted_key=-2
        )
        table.insert(img_idx, idx_tensor)

        # Get true boxes num array
        ground_truth_num = tf.TensorArray(tf.int32, size=tf.math.reduce_max(idx) + 1, dynamic_size=True, clear_after_read=False)
        for i in tf.range(tf.math.reduce_max(idx) + 1):
            ground_truth_num = ground_truth_num.write(i, tf.zeros(tf.math.reduce_max(count), dtype=tf.int32))

        # sort by confidence score
        detections = tf.gather(detections, tf.argsort(detections[..., 2], direction='DESCENDING'))
        true_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())
        false_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())

        detections_size = tf.shape(detections)[0]
        detection_idx = tf.constant(0, dtype=tf.int32)
        for detection in detections:
            # tf.print('progressing of detection: ', detection_idx, ' / ', detections_size)
            # tf.print('detection_img_idx: ', detection[0])
            # tf.print('detection_confidence: ', detection[2])

            ground_truth_img = tf.gather(ground_truths, tf.reshape(tf.where(ground_truths[..., 0] == detection[0]),
                                                                   shape=(-1,)))
            # tf.print('ground_truth_img: ', tf.shape(ground_truth_img)[0])

            best_iou = tf.TensorArray(tf.float32, size=1, element_shape=(1,), clear_after_read=False)
            best_gt_idx = tf.TensorArray(tf.int32, size=1, element_shape=(), clear_after_read=False)

            gt_idx = tf.constant(0, dtype=tf.int32)
            for gt_img in ground_truth_img:
                iou = intersection_over_union(detection[3:], gt_img[3:])

                if iou > best_iou.read(0):
                    best_iou = best_iou.write(0, iou)
                    best_gt_idx = best_gt_idx.write(0, gt_idx)

                gt_idx += 1

            if best_iou.read(0) > iou_threshold:
                # Get current detections img_idx
                cur_det_img_idx = tf.cast(detection[0], dtype=tf.int32)

                # Get row idx of ground_truth_num array
                gt_row_idx = table.lookup(cur_det_img_idx)

                # Get 'current img ground_truth_num tensor'
                cur_gt_num_tensor = ground_truth_num.read(gt_row_idx)

                # Get idx of current best ground truth
                cur_best_gt_idx = best_gt_idx.read(0)

                if cur_gt_num_tensor[cur_best_gt_idx] == 0:
                    true_positive = true_positive.write(detection_idx, 1)

                    # change cur_gt_num_tensor[cur_best_gt_idx] to 1
                    cur_gt_num_tensor = change_tensor(cur_gt_num_tensor, cur_best_gt_idx)

                    # update ground_truth_num array
                    ground_truth_num = ground_truth_num.write(gt_row_idx, cur_gt_num_tensor)

                else:
                    false_positive = false_positive.write(detection_idx, 1)

            # if IOU is lower then the detection is a false positive
            else:
                false_positive = false_positive.write(detection_idx, 1)

            # ground_truth_img.close()
            best_iou.close()
            best_gt_idx.close()
            detection_idx += 1

        # Compute the cumulative sum of the tensor
        tp_cumsum = tf.math.cumsum(true_positive.stack(), axis=0)
        fp_cumsum = tf.math.cumsum(false_positive.stack(), axis=0)

        # Calculate recalls and precisions
        recalls = tf.math.divide(tp_cumsum, (total_true_boxes + epsilon))
        precisions = tf.math.divide(tp_cumsum, (tp_cumsum + fp_cumsum + epsilon))

        # Append start point value of precision-recall graph
        precisions = tf.concat([tf.constant([1], dtype=tf.float32), precisions], axis=0)
        recalls = tf.concat([tf.constant([0], dtype=tf.float32), recalls], axis=0)
        # tf.print(precisions)
        # tf.print(recalls)

        # Calculate area of precision-recall graph
        average_precision_value = tf.py_function(func=np.trapz,
                                                 inp=[precisions, recalls],
                                                 Tout=tf.float32)
        average_precisions = average_precisions.write(average_precisions.size(), average_precision_value)
        # tf.print('average precision: ', average_precision_value)

        ground_truth_num.close()
        true_positive.close()
        false_positive.close()

    # tf.print(average_precisions.stack())
    tf.print('mAP: ', tf.math.reduce_mean(average_precisions.stack()))
    return tf.math.reduce_mean(average_precisions.stack())


class MeanAveragePrecision:
    def __init__(self, num_classes, num_boxes=2):
        self.all_true_boxes_variable = tf.Variable([[-1, -1, -1, -1, -1, -1, -1]], dtype=tf.float32, shape=tf.TensorShape((None, 7)))
        self.all_pred_boxes_variable = tf.Variable([[-1, -1, -1, -1, -1, -1, -1]], dtype=tf.float32, shape=tf.TensorShape((None, 7)))
        self.img_idx = tf.Variable([0], dtype=tf.float32)
        self._num_classes = num_classes
        self._num_boxes = num_boxes

    def reset_states(self):
        self.img_idx.assign([0])

    def update_state(self, y_true, y_pred):
        true_boxes = decode_predictions(y_true, self._num_classes, self._num_boxes)
        pred_boxes = decode_predictions(y_pred, self._num_classes, self._num_boxes)

        for idx in tf.range(tf.shape(y_true)[0]):
            pred_nms = non_max_suppression(pred_boxes[idx], iou_threshold=0.5, conf_threshold=0.4)
            pred_img_idx = tf.zeros([tf.shape(pred_nms)[0], 1], tf.float32) + self.img_idx
            pred_concat = tf.concat([pred_img_idx, pred_nms], axis=1)

            # true_bbox = tf.gather(true_boxes[idx], tf.reshape(tf.where(true_boxes[idx][..., 1] > 0.4), shape=(-1,)))
            true_nms = non_max_suppression(true_boxes[idx], iou_threshold=0.5, conf_threshold=0.4)
            true_img_idx = tf.zeros([tf.shape(true_nms)[0], 1], tf.float32) + self.img_idx
            true_concat = tf.concat([true_img_idx, true_nms], axis=1)

            if self.img_idx == 0.:
                self.all_true_boxes_variable.assign(true_concat)
                self.all_pred_boxes_variable.assign(pred_concat)
            else:
                self.all_true_boxes_variable.assign(tf.concat([self.all_true_boxes_variable, true_concat], axis=0))
                self.all_pred_boxes_variable.assign(tf.concat([self.all_pred_boxes_variable, pred_concat], axis=0))

            self.img_idx.assign_add([1])

    def result(self):
        tf.print('all true bboxes: ', tf.shape(self.all_true_boxes_variable)[0])
        tf.print('all pred bboxes: ', tf.shape(self.all_pred_boxes_variable)[0])
        return mean_average_precision(self.all_true_boxes_variable, self.all_pred_boxes_variable, self._num_classes)


def mean_average_precision_numpy(true_boxes, pred_boxes, num_classes, iou_threshold=0.5):
    """Calculates mean average precision

    Arguments:
        true_boxes (Numpy Array): Numpy Array of all bboxes with all images (None, 7), specified as [img_idx, class_idx, confidence_score, cx, cy, w, h]
        pred_boxes (Numpy Array): Similar as true_boxes
        num_classes (int): number of classes
        iou_threshold (float): threshold where predicted boxes is correct

    Returns:
        Float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in np.arange(num_classes, dtype=np.float32):
        print(f'Calculating AP: {c} / {num_classes}')
        
        # detections, ground_truths variables in specific class
        detections = np.take(pred_boxes, np.where(pred_boxes[..., 1] == c)[0], axis=0)
        ground_truths = np.take(true_boxes, np.where(true_boxes[..., 1] == c)[0], axis=0)

        # If none exists for this class then we can safely skip
        total_true_boxes = tf.cast(tf.shape(ground_truths)[0], dtype=tf.float32)
        if ground_truths.shape[0] == 0:
            average_precisions.append(0.)
            continue
        
        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_boxes = {0:3, 1:5}
        amount_boxes = Counter(gt[0] for gt in ground_truths)
        
        for key, val in amount_boxes.items():
            amount_boxes[key] = np.zeros(val)
        # amount_boxes = {0: torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}

        # sort by confidence score
        detections = np.take(detections, np.argsort(-detections[..., 2]), axis=0)
        true_positive = np.zeros(len(detections))
        false_positive = np.zeros(len(detections))

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = np.take(ground_truths, tf.where(ground_truths[..., 0] == detection[0])[0], axis=0)
            best_iou = 0
            best_gt_idx = 0
            
            for gt_idx, gt_img in enumerate(ground_truth_img):
                iou = intersection_over_union_numpy(detection[3:], gt_img[3:])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

                gt_idx += 1

            if best_iou > iou_threshold:
                if amount_boxes[detection[0]][best_gt_idx] == 0:
                    true_positive[detection_idx] = 1
                    amount_boxes[detection[0]][best_gt_idx] = 1
                else:
                    false_positive[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        # Compute the cumulative sum of the tensor
        tp_cumsum = np.cumsum(true_positive, axis=0)
        fp_cumsum = np.cumsum(false_positive, axis=0)

        # Calculate recalls and precisions
        recalls = np.divide(tp_cumsum, (total_true_boxes + epsilon))
        precisions = np.divide(tp_cumsum, (tp_cumsum + fp_cumsum + epsilon))

        # Append start point value of precision-recall graph
        precisions = np.concatenate([np.array([1], dtype=np.float32), precisions], axis=0)
        recalls = np.concatenate([np.array([0], dtype=np.float32), recalls], axis=0)

        # Calculate area of precision-recall graph
        average_precision_value = np.trapz(precisions, recalls)
        average_precisions.append(average_precision_value)

    print(f'mAP: {np.mean(average_precisions)}')
    return np.mean(average_precisions)


class MeanAveragePrecisionNumpy:
    def __init__(self, num_classes, num_boxes=2):
        self.all_true_boxes_variable = np.zeros((0, 7), dtype=np.float32)
        self.all_pred_boxes_variable = np.zeros((0, 7), dtype=np.float32)
        self.img_idx = 0.
        self._num_classes = num_classes
        self._num_boxes = num_boxes

    def reset_states(self):
        self.img_idx = 0.

    def update_state(self, y_true, y_pred):
        true_boxes = decode_predictions_numpy(y_true, self._num_classes, self._num_boxes)
        pred_boxes = decode_predictions_numpy(y_pred, self._num_classes, self._num_boxes)

        for idx in np.arange(y_true.shape[0]):
            pred_nms = non_max_suppression_numpy(pred_boxes[idx], iou_threshold=0.5, conf_threshold=0.4)
            pred_img_idx = np.zeros([pred_nms.shape[0], 1], np.float32) + self.img_idx
            pred_concat = np.concatenate([pred_img_idx, pred_nms], axis=1)

            true_nms = non_max_suppression_numpy(true_boxes[idx], iou_threshold=0.5, conf_threshold=0.4)
            true_img_idx = np.zeros([true_nms.shape[0], 1], np.float32) + self.img_idx
            true_concat = np.concatenate([true_img_idx, true_nms], axis=1)

            self.all_true_boxes_variable = np.append(self.all_true_boxes_variable, true_concat, axis=0)
            self.all_pred_boxes_variable = np.append(self.all_pred_boxes_variable, pred_concat, axis=0)

            self.img_idx += 1

    def result(self):
        print('all true bboxes: ', self.all_true_boxes_variable.shape[0])
        print('all pred bboxes: ', self.all_pred_boxes_variable.shape[0])
        return mean_average_precision_numpy(self.all_true_boxes_variable, self.all_pred_boxes_variable, self._num_classes)


def get_tagged_img(img, boxes, names_path):
    """tagging result on img

    Arguments:
        img (Numpy Array): Image array
        boxes (Tensor): boxes after performing NMS (None, 6)
        names_path (String): path of label names file

    Returns:
        Numpy Array: tagged image array
    """
    
    if not tf.is_tensor(boxes):
        boxes = tf.cast(boxes, dtype=tf.float32)

    width = img.shape[1]
    height = img.shape[0]
    
    with open(names_path, 'r') as f:
        class_name_list = f.readlines()
    class_name_list = [x.strip() for x in class_name_list]
    
    for box in boxes:
        class_name = class_name_list[int(box[0])]
        confidence_score = box[1]
        x = box[2]
        y = box[3]
        w = box[4]
        h = box[5]
        xmin = int((x - (w / 2)) * width)
        ymin = int((y - (h / 2)) * height)
        xmax = int((x + (w / 2)) * width)
        ymax = int((y + (h / 2)) * height)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
        img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=(0, 255, 0))

    return img


def get_grid_tagged_img(img, boxes, names_path):
    """tagging result on img with grid cell

    Arguments:
        img (Numpy Array): Image array
        boxes (Tensor): boxes after performing NMS (None, 6)
        names_path (String): path of label names file

    Returns:
        Numpy Array: tagged image array
    """
    
    if not tf.is_tensor(boxes):
        boxes = tf.cast(boxes, dtype=tf.float32)
    
    width = img.shape[1]
    height = img.shape[0]
    
    with open(names_path, 'r') as f:
        class_name_list = f.readlines()
    class_name_list = [x.strip() for x in class_name_list]
    
    for box in boxes:
        class_name = class_name_list[int(box[0])]
        confidence_score = box[1]
        x = box[2]
        y = box[3]
        w = box[4]
        h = box[5]
        xmin = int((x - (w / 2)) * width)
        ymin = int((y - (h / 2)) * height)
        xmax = int((x + (w / 2)) * width)
        ymax = int((y + (h / 2)) * height)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
        img = cv2.circle(img, (int(x*width), int(y*height)), radius=2, color=(0, 0, 255))
        img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=(0, 255, 0))
    
    # Draw grid cell
    for idx in range(6):
        a = int(448 * ((idx + 1) / 7.))
        img = cv2.line(img, (a, 0), (a, height), color=(255, 0, 255))
        img = cv2.line(img, (0, a), (width, a), color=(255, 0, 255))

    return img


if __name__ == "__main__":
    num_classes = 3
    num_boxes = 2
    y_true = np.zeros(shape=(1, 7, 7, (num_classes + (5*num_boxes))), dtype=np.float32)
    y_true[:, 0, 0, 0] = 1 # class
    y_true[:, 0, 0, num_classes] = 1 # confidence1
    y_true[:, 0, 0, num_classes+1:num_classes+5] = [0.5, 0.5, 0.1, 0.1] # box1
    
    y_true[:, 3, 3, 1] = 1 # class
    y_true[:, 3, 3, num_classes] = 1 # confidence1
    y_true[:, 3, 3, num_classes+1:num_classes+5] = [0.5, 0.5, 0.1, 0.1] # box1
    
    y_true[:, 6, 6, 2] = 1 # class
    y_true[:, 6, 6, num_classes] = 1 # confidence1
    y_true[:, 6, 6, num_classes+1:num_classes+5] = [0.5, 0.5, 0.1, 0.1] # box1
    
    y_true = tf.cast(y_true, tf.float32)
    # print(y_true)
    
    y_pred = np.zeros(shape=(1, 7, 7, (num_classes + (5*num_boxes))), dtype=np.float32)
    y_pred[:, 0, 0, :num_classes] = [0.8, 0.5, 0.1] # class
    y_pred[:, 0, 0, num_classes] = 0.6 # confidence1
    y_pred[:, 0, 0, num_classes+1:num_classes+5] = [0.49, 0.49, 0.1, 0.1] # box1
    y_pred[:, 0, 0, num_classes+5] = 0.2 # confidence2
    y_pred[:, 0, 0, num_classes+6:num_classes+10] = [0.45, 0.45, 0.1, 0.1] # box2
    
    y_pred[:, 3, 3, :num_classes] = [0.2, 0.8, 0.1] # class
    y_pred[:, 3, 3, num_classes] = 0.1 # confidence1
    y_pred[:, 3, 3, num_classes+1:num_classes+5] = [0.45, 0.45, 0.1, 0.1] # box1
    y_pred[:, 3, 3, num_classes+5] = 0.9 # confidence2
    y_pred[:, 3, 3, num_classes+6:num_classes+10] = [0.49, 0.49, 0.1, 0.1] # box2
    
    y_pred[:, 6, 6, :num_classes] = [0.1, 0.5, 0.8] # class
    y_pred[:, 6, 6, num_classes] = 0.6 # confidence1
    y_pred[:, 6, 6, num_classes+1:num_classes+5] = [0.49, 0.49, 0.1, 0.1] # box1
    y_pred[:, 6, 6, num_classes+5] = 0.2 # confidence2
    y_pred[:, 6, 6, num_classes+6:num_classes+10] = [0.45, 0.45, 0.1, 0.1] # box2
    
    y_pred = tf.cast(y_pred, tf.float32)
    # print(y_pred)
    
    decode_pred = decode_predictions(y_true, num_classes, num_boxes)
    nms = non_max_suppression(decode_pred[0])
    print(nms)
    
    decode_pred_np = decode_predictions_numpy(np.array(y_true, np.float32), num_classes, num_boxes)
    nms_np = non_max_suppression_numpy(decode_pred_np[0])
    print(nms_np)
    
    