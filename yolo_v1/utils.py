from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
import tensorflow as tf
import cv2
import time


@tf.function
def intersection_over_union(bbox_true, bbox_pred):
    """
    Calculates intersection-over-union

    Parameters:
        bbox_true (Tensor): true bbox       (batch, S, S, 4(x,y,w,h))
        bbox_pred (Tensor): prediction bbox (batch, S, S, 4(x,y,w,h)

    Returns:
        Tensor: Tensor of iou (batch, S, S, 1)
    """
    assert tf.is_tensor(bbox_true) and tf.is_tensor(bbox_pred)

    box1_xmin = (bbox_true[..., 0:1] - bbox_true[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymin = (bbox_true[..., 1:2] - bbox_true[..., 3:4]) / 2. # (batch, S, S, 1)
    box1_xmax = (bbox_true[..., 0:1] + bbox_true[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymax = (bbox_true[..., 1:2] + bbox_true[..., 3:4]) / 2. # (batch, S, S, 1)

    box2_xmin = (bbox_pred[..., 0:1] - bbox_pred[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymin = (bbox_pred[..., 1:2] - bbox_pred[..., 3:4]) / 2. # (batch, S, S, 1)
    box2_xmax = (bbox_pred[..., 0:1] + bbox_pred[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymax = (bbox_pred[..., 1:2] + bbox_pred[..., 3:4]) / 2. # (batch, S, S, 1)

    inter_xmin = tf.math.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
    inter_ymin = tf.math.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
    inter_xmax = tf.math.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
    inter_ymax = tf.math.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)

    inter_area = tf.clip_by_value((inter_xmax - inter_xmin), 0, 1) * tf.clip_by_value((inter_ymax - inter_ymin), 0, 1) # (batch, S, S, 1)
    box1_area = tf.math.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = tf.math.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)


def intersection_over_union_numpy(bbox_true, bbox_pred):
    """
    Calculates intersection-over-union

    Parameters:
        bbox_true (Numpy Array): true bbox       (batch, S, S, 4(x,y,w,h))
        bbox_pred (Numpy Array): prediction bbox (batch, S, S, 4(x,y,w,h)

    Returns:
        Numpy Array: Numpy Array of iou (batch, S, S, 1)
    """

    box1_xmin = (bbox_true[..., 0:1] - bbox_true[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymin = (bbox_true[..., 1:2] - bbox_true[..., 3:4]) / 2. # (batch, S, S, 1)
    box1_xmax = (bbox_true[..., 0:1] + bbox_true[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymax = (bbox_true[..., 1:2] + bbox_true[..., 3:4]) / 2. # (batch, S, S, 1)

    box2_xmin = (bbox_pred[..., 0:1] - bbox_pred[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymin = (bbox_pred[..., 1:2] - bbox_pred[..., 3:4]) / 2. # (batch, S, S, 1)
    box2_xmax = (bbox_pred[..., 0:1] + bbox_pred[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymax = (bbox_pred[..., 1:2] + bbox_pred[..., 3:4]) / 2. # (batch, S, S, 1)

    inter_xmin = np.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
    inter_ymin = np.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
    inter_xmax = np.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
    inter_ymax = np.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)

    inter_area = np.clip((inter_xmax - inter_xmin), 0, 1) * np.clip((inter_ymax - inter_ymin), 0, 1) # (batch, S, S, 1)
    box1_area = np.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = np.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)

@tf.function
def get_all_bboxes(out, S=7, C=20):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.

    Parameters:
        out (Tensor): predictions or true_labels (batch, S, S, (B*5)+C)

    Returns:
        Tensor: Tensor of bboxes (batch, S*S, 6(class_index, confidence, x, y, w, h))
    """
    assert tf.is_tensor(out)

    bbox1_start_index = C + 1
    bbox1_confidence_index = C
    bbox2_start_index = C + 1 + 5
    bbox2_confidence_index = C + 5

    bboxes1 = out[..., bbox1_start_index:bbox1_start_index+4]  # (batch, S, S, 4)
    bboxes2 = out[..., bbox2_start_index:bbox2_start_index+4] # (batch, S, S, 4)
    confidences = tf.concat([out[..., bbox1_confidence_index:bbox1_confidence_index+1],
                             out[..., bbox2_confidence_index:bbox2_confidence_index+1]], axis=-1) # (batch, S, S, 2)

    # Get best box index
    best_box_index = tf.math.argmax(confidences, axis=-1) # (batch, S, S)
    best_box_index = tf.expand_dims(best_box_index, axis=-1) # (batch, S, S, 1)
    best_box_index = tf.cast(best_box_index, dtype=np.float32)

    # Get best boxes
    best_boxes = (1 - best_box_index) * bboxes1 + best_box_index * bboxes2 # (batch, S, S, 4)

    # Get cell indexes array
    base_arr = tf.map_fn(fn=lambda x: tf.range(x, x + S), elems=tf.zeros(S))
    x_cell_indexes = tf.reshape(base_arr, shape=(S, S, 1)) # (S, S, 1)

    y_cell_indexes = tf.transpose(base_arr)
    y_cell_indexes = tf.reshape(y_cell_indexes, shape=(S, S, 1)) # (S, S, 1)

    # Convert x, y ratios to YOLO ratios
    x = 1 / S * (best_boxes[..., :1] + x_cell_indexes) # (batch, S, S, 1)
    y = 1 / S * (best_boxes[..., 1:2] + y_cell_indexes) # (batch, S, S, 1)

    # Get class indexes
    class_indexes = tf.math.argmax(out[..., :C], axis=-1) # (batch, S, S)
    class_indexes = tf.expand_dims(class_indexes, axis=-1) # (batch, S, S, 1)
    class_indexes = tf.cast(class_indexes, dtype=np.float32)

    # Get best confidences
    best_confidences = (1 - best_box_index) * out[..., bbox1_confidence_index:bbox1_confidence_index+1] + \
                       best_box_index * out[..., bbox2_confidence_index:bbox2_confidence_index+1] # (batch, S, S, 1)

    # Get converted bboxes
    converted_bboxes = tf.concat([x, y, best_boxes[..., 2:4]], axis=-1) # (batch, S, S, 4)

    # Concatenate result
    converted_out = tf.concat([class_indexes, best_confidences, converted_bboxes], axis=-1) # (batch, S, S, 6)

    # Get all bboxes
    converted_out = tf.reshape(converted_out, shape=(-1, S * S, 6)) # (batch, S*S, 6)

    return converted_out # (batch, S*S, 6)


def get_all_bboxes_numpy(predictions, S=7, C=20):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.

    Parameters:
        predictions (Numpy Array): predictions or true_labels (batch, S, S, (B*5)+C)

    Returns:
        Numpy Array: numpy array of bboxes (batch, S*S, 6(class_index, confidence, x, y, w, h))
    """

    bbox1_start_index = C + 1
    bbox1_confidence_index = C
    bbox2_start_index = C + 1 + 5
    bbox2_confidence_index = C + 5

    bboxes1 = predictions[..., bbox1_start_index:bbox1_start_index + 4]  # (batch, S, S, 4)
    bboxes2 = predictions[..., bbox2_start_index:bbox2_start_index + 4]  # (batch, S, S, 4)
    confidences = np.concatenate((predictions[..., bbox1_confidence_index:bbox1_confidence_index + 1],
                                  predictions[..., bbox2_confidence_index:bbox2_confidence_index + 1]),
                                 axis=-1)  # (batch, S, S, 2)

    # Get best box index
    best_box_index = np.argmax(confidences, axis=-1)  # (batch, S, S)
    best_box_index = np.expand_dims(best_box_index, axis=-1)  # (batch, S, S, 1)
    best_box_index = best_box_index.astype(np.float32)

    # Get best boxes
    best_boxes = (1 - best_box_index) * bboxes1 + best_box_index * bboxes2  # (batch, S, S, 4)

    # Get cell indexes array
    base_arr = np.arange(S).reshape((1, -1)).repeat(S, axis=0)
    x_cell_indexes = np.expand_dims(base_arr, axis=-1)  # (S, S, 1)

    y_cell_indexes = np.transpose(base_arr)
    y_cell_indexes = np.expand_dims(y_cell_indexes, axis=-1)  # (S, S, 1)

    # Convert x, y ratios to YOLO ratios
    x = 1 / S * (best_boxes[..., :1] + x_cell_indexes)  # (batch, S, S, 1)
    y = 1 / S * (best_boxes[..., 1:2] + y_cell_indexes)  # (batch, S, S, 1)

    # Get class indexes
    class_indexes = np.argmax(predictions[..., :C], axis=-1)  # (batch, S, S)
    class_indexes = np.expand_dims(class_indexes, axis=-1)  # (batch, S, S, 1)
    class_indexes = class_indexes.astype(np.float32)

    # Get best confidences
    best_confidences = (1 - best_box_index) * predictions[..., bbox1_confidence_index:bbox1_confidence_index + 1] + \
                       best_box_index * predictions[...,
                                        bbox2_confidence_index:bbox2_confidence_index + 1]  # (batch, S, S, 1)

    # Get converted bboxes
    converted_bboxes = np.concatenate((x, y, best_boxes[..., 2:4]), axis=-1)  # (batch, S, S, 4)

    # Concatenate result
    converted_out = np.concatenate((class_indexes, best_confidences, converted_bboxes), axis=-1)  # (batch, S, S, 6)

    # Get all bboxes
    converted_out = np.reshape(converted_out, (-1, S * S, 6))  # (batch, S*S, 6)

    return converted_out # (batch, S*S, 6)


@tf.function
def non_max_suppression(bboxes, iou_threshold=0.5, threshold=0.4):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (Tensor): tensor of all bboxes with each grid (S*S, 6)
        specified as [class_idx, confidence_score, x, y, w, h]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes

    Returns:
        Tensor: bboxes after performing NMS given a specific IoU threshold (None, 6)
    """
    assert tf.is_tensor(bboxes)

    # bboxes smaller than the threshold are removed
    # bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for bbox in bboxes:
        if bbox[1] > threshold:
            bboxes_array = bboxes_array.write(bboxes_array.size(), bbox)

    # sort descending by confidence score
    # bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_array = bboxes_array.scatter(tf.argsort(bboxes_array.stack()[:, 1], direction='DESCENDING'),
                                        bboxes_array.stack())

    # get bboxes after nms
    # bboxes_after_nms = []
    bboxes_after_nms = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    # loop variables
    loop_vars = (bboxes_array, bboxes_after_nms)

    # loop condition
    loop_cond = lambda i, j: not(tf.less(i.size(), 1))

    # loop body
    def loop_body(i, j):
        chosen_box = i.read(0)
        tmp_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for idx in tf.range(1, i.size()):
            bbox = i.read(idx)
            if bbox[0] != chosen_box[0] or intersection_over_union(
                    chosen_box[2:], bbox[2:]
            ) < iou_threshold:
                tmp_array = tmp_array.write(tmp_array.size(), bbox)
        i = tmp_array

        j = j.write(j.size(), chosen_box)

        return i, j

    _, bboxes_after_nms = tf.while_loop(loop_cond, loop_body, loop_vars)

    # # while bboxes:
    # while bboxes_array.size():
    #     # chosen_box = bboxes.pop(0)
    #     chosen_box = bboxes_array.read(0)
    #
    #     # bboxes = [
    #     #     box
    #     #     for box in bboxes
    #     #     if box[0] != chosen_box[0]
    #     #     or intersection_over_union(
    #     #         tf.cast(chosen_box[2:], dtype=np.float32),
    #     #         tf.cast(box[2:], dtype=np.float32))
    #     #        < iou_threshold
    #     # ]
    #     tmp_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    #     for idx in range(1, bboxes_array.size()):
    #         bbox = tmp_array.read(idx)
    #         if bbox[0] != chosen_box[0] or intersection_over_union(
    #             chosen_box[2:], bbox[2:]
    #         ) < iou_threshold:
    #             tmp_array = tmp_array.write(tmp_array.size(), bbox)
    #     bboxes_array = tmp_array
    #
    #     # bboxes_after_nms.append(chosen_box)
    #     bboxes_after_nms = bboxes_after_nms.write(bboxes_after_nms.size(), chosen_box)

    return bboxes_after_nms.stack()


@tf.function
def non_max_suppression_2(bboxes, iou_threshold=0.5, threshold=0.4):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (Tensor): tensor of all bboxes with each grid (S*S, 6)
        specified as [class_idx, confidence_score, x, y, w, h]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes

    Returns:
        Tensor: bboxes after performing NMS given a specific IoU threshold (None, 6)
    """
    assert tf.is_tensor(bboxes)

    # bboxes smaller than the threshold are removed
    bboxes_new = tf.gather(bboxes, tf.reshape(tf.where(bboxes[..., 1] > threshold), shape=(-1,)))

    # sort descending by confidence score
    bboxes_new = tf.gather(bboxes_new, tf.argsort(bboxes_new[..., 1], direction='DESCENDING'))

    # get bboxes after nms
    bboxes_after_nms = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    while not(tf.less(tf.shape(bboxes_new)[0], 1)):
        chosen_box = bboxes_new[0]
        tmp_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for idx in tf.range(1, tf.shape(bboxes_new)[0]):
            bbox = bboxes_new[idx]
            if bbox[0] != chosen_box[0] or intersection_over_union(chosen_box[2:], bbox[2:]) < iou_threshold:
                tmp_array = tmp_array.write(tmp_array.size(), bbox)
        bboxes_new = tmp_array.stack()

        bboxes_after_nms = bboxes_after_nms.write(bboxes_after_nms.size(), chosen_box)

    return bboxes_after_nms.stack()


def non_max_suppression_numpy(bboxes, iou_threshold=0.5, threshold=0.4):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (Numpy Array): numpy array of all bboxes with each grid (S*S, 6)
        specified as [class_idx, confidence_score, x, y, w, h]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes

    Returns:
        Numpy Array: bboxes after performing NMS given a specific IoU threshold (None, 6)
    """

    # bboxes smaller than the threshold are removed
    bboxes_new = np.take(bboxes, np.where(bboxes[..., 1] > threshold)[0], axis=0)

    # sort descending by confidence score
    bboxes_new = np.take(bboxes_new, np.argsort(-bboxes_new[..., 1]), axis=0)

    # get bboxes after nms
    bboxes_after_nms = np.empty(shape=(0, 6))

    while not(np.less(bboxes_new.shape[0], 1)):
        chosen_box = np.expand_dims(bboxes_new[0], axis=0)
        tmp_array = np.empty(shape=(0, 6))
        for idx in range(1, bboxes_new.shape[0]):
            bbox = np.expand_dims(bboxes_new[idx], axis=0)
            if bbox[0][0] != chosen_box[0][0] or intersection_over_union_numpy(chosen_box[..., 2:], bbox[..., 2:]) < iou_threshold:
                tmp_array = np.append(tmp_array, bbox, axis=0)
        bboxes_new = tmp_array

        bboxes_after_nms = np.append(bboxes_after_nms, chosen_box, axis=0)

    return bboxes_after_nms


@tf.function(input_signature=[tf.TensorSpec(shape=[None, 7]), tf.TensorSpec(shape=[None, 7])])
def mean_average_precision(true_bboxes, pred_bboxes, iou_threshold=0.5, num_classes=20):
    print('mean average precision tracing')
    """
    Calculates mean average precision

    Parameters:
        true_bboxes (Tensor): Tensor of all bboxes with all images (None, 7)
        specified as [img_idx, class_idx, confidence_score, x, y, w, h]
        pred_bboxes (Tensor): Similar as true_bboxes
        iou_threshold (float): threshold where predicted bboxes is correct
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """
    assert tf.is_tensor(true_bboxes) and tf.is_tensor(pred_bboxes)

    # list storing all AP for respective classes
    average_precisions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    # used for numerical stability later on
    epsilon = 1e-6

    for c in tf.range(num_classes, dtype=tf.float32):
        tf.print('calculating AP: ', c, ' / ', num_classes)
        # detections, ground_truths variables in specific class
        detections = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        ground_truths = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_bboxes:
            if detection[1] == c:
                detections = detections.write(detections.size(), detection)

        for true_box in true_bboxes:
            if true_box[1] == c:
                ground_truths = ground_truths.write(ground_truths.size(), true_box)

        # If none exists for this class then we can safely skip
        # total_true_bboxes = len(ground_truths)
        total_true_bboxes = tf.cast(ground_truths.size(), dtype=tf.float32)
        if total_true_bboxes == 0.:
            average_precisions = average_precisions.write(average_precisions.size(),
                                                          tf.constant(0, dtype=tf.float32))
            continue

        # tf.print(c, ' class ground truths size: ', ground_truths.size())
        # tf.print(c, ' class detections size: ', detections.size())

        # Get the number of true bbox by image index
        img_idx, idx, count = tf.unique_with_counts(ground_truths.stack()[..., 0])
        img_idx = tf.cast(img_idx, dtype=tf.int32)

        # Convert idx to idx tensor for find num of true bbox by img idx
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
        # tf.print('table: ', table.export())
        # tf.print('table[0]: ', table.lookup(tf.constant([0], dtype=tf.int32)))

        # Get true bbox num array
        ground_truth_num = tf.TensorArray(tf.int32, size=tf.math.reduce_max(idx) + 1, dynamic_size=True, clear_after_read=False)
        for i in tf.range(tf.math.reduce_max(idx) + 1):
            ground_truth_num = ground_truth_num.write(i, tf.zeros(tf.math.reduce_max(count), dtype=tf.int32))

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        # amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        # for key, val in amount_bboxes.items():
        #     amount_bboxes[key] = np.zeros(val)

        # sort by bbox confidence score
        # detections.sort(key=lambda x: x[2], reverse=True)
        # TP = np.zeros((len(detections)))
        # FP = np.zeros((len(detections)))
        detections = detections.gather(tf.argsort(detections.stack()[..., 2], direction='DESCENDING'))
        true_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())
        false_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())

        detections_size = tf.shape(detections)[0]
        detection_idx = tf.constant(0, dtype=tf.int32)
        for detection in detections:
            # tf.print('progressing of detection: ', detection_idx, ' / ', detections_size)
            # Only take out the ground_truths that have the same
            # training idx as detection
            # ground_truth_img = [
            #     bbox for bbox in ground_truths if bbox[0] == detection[0]
            # ]
            # tf.print('detection_img_idx: ', detection[0])
            # tf.print('detection_confidence: ', detection[2])

            ground_truth_img = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for ground_truth_idx in tf.range(ground_truths.size()):
                if ground_truths.read(ground_truth_idx)[0] == detection[0]:
                    ground_truth_img = ground_truth_img.write(ground_truth_img.size(), ground_truths.read(ground_truth_idx))
            # tf.print('ground_truth_img: ', ground_truth_img.size())

            best_iou = tf.TensorArray(tf.float32, size=1, element_shape=(1,), clear_after_read=False)
            best_gt_idx = tf.TensorArray(tf.int32, size=1, element_shape=(), clear_after_read=False)

            for idx in tf.range(ground_truth_img.size()):
                # iou = intersection_over_union(
                #     tf.cast(detection[3:], dtype=tf.float32),
                #     tf.cast(gt[3:], dtype=tf.float32)
                # )

                iou = intersection_over_union(
                    detection[3:],
                    ground_truth_img.read(idx)[3:]
                )

                if iou > best_iou.read(0):
                    best_iou = best_iou.write(0, iou)
                    best_gt_idx = best_gt_idx.write(0, idx)

            if best_iou.read(0) > iou_threshold:
                # # only detect ground truth detection once
                # if amount_bboxes[detection[0]][best_gt_idx] == 0:
                #     # true positive and add this bounding box to seen
                #     TP[detection_idx] = 1
                #     amount_bboxes[detection[0]][best_gt_idx] = 1
                # else:
                #     FP[detection_idx] = 1

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

            ground_truth_img.close()
            best_iou.close()
            best_gt_idx.close()
            detection_idx += 1

        # Compute the cumulative sum of the tensor
        # tf.print("TP: ", true_positive.stack())
        # tf.print("FP: ", false_positive.stack())
        tp_cumsum = tf.math.cumsum(true_positive.stack(), axis=0)
        fp_cumsum = tf.math.cumsum(false_positive.stack(), axis=0)

        # Calculate recalls and precisions
        recalls = tf.math.divide(tp_cumsum, (total_true_bboxes + epsilon))
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

        ground_truths.close()
        ground_truth_num.close()
        true_positive.close()
        false_positive.close()

        # TP_cumsum = np.cumsum(TP, axis=0)
        # FP_cumsum = np.cumsum(FP, axis=0)
        # recalls = TP_cumsum / (total_true_bboxes + epsilon)
        # precisions = np.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        # precisions = np.concatenate(([1.], precisions), axis=0)
        # recalls = np.concatenate(([0.], recalls), axis=0)
        # # tf.math.trapz for numerical integration
        # average_precisions.append(np.trapz(precisions, recalls))
    # tf.print(average_precisions.stack())
    tf.print('mAP: ', tf.math.reduce_mean(average_precisions.stack()))
    return tf.math.reduce_mean(average_precisions.stack())


@tf.function
def change_tensor(tensor_1d, idx_col):
    """
    change the value of a specific column in a tensor to 1

    Parameters:
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


@tf.function(input_signature=[tf.TensorSpec(shape=[None, 7]), tf.TensorSpec(shape=[None, 7])])
def mean_average_precision_2(true_bboxes, pred_bboxes, iou_threshold=0.5, num_classes=20):
    print('mean average precision tracing')
    """
    Calculates mean average precision

    Parameters:
        true_bboxes (Tensor): Tensor of all bboxes with all images (None, 7)
        specified as [img_idx, class_idx, confidence_score, x, y, w, h]
        pred_bboxes (Tensor): Similar as true_bboxes
        iou_threshold (float): threshold where predicted bboxes is correct
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """
    assert tf.is_tensor(true_bboxes) and tf.is_tensor(pred_bboxes)

    # list storing all AP for respective classes
    average_precisions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    # used for numerical stability later on
    epsilon = 1e-6

    for c in tf.range(num_classes, dtype=tf.float32):
        tf.print('calculating AP: ', c, ' / ', num_classes)
        # detections, ground_truths variables in specific class
        detections = tf.gather(pred_bboxes, tf.reshape(tf.where(pred_bboxes[..., 1] == c), shape=(-1,)))
        ground_truths = tf.gather(true_bboxes, tf.reshape(tf.where(true_bboxes[..., 1] == c), shape=(-1,)))

        # If none exists for this class then we can safely skip
        total_true_bboxes = tf.cast(tf.shape(ground_truths)[0], dtype=tf.float32)
        if total_true_bboxes == 0.:
            average_precisions = average_precisions.write(average_precisions.size(),
                                                          tf.constant(0, dtype=tf.float32))
            continue

        # tf.print(c, ' class ground truths size: ', tf.shape(ground_truths)[0])
        # tf.print(c, ' class detections size: ', tf.shape(detections)[0])

        # Get the number of true bbox by image index
        img_idx, idx, count = tf.unique_with_counts(ground_truths[..., 0])
        img_idx = tf.cast(img_idx, dtype=tf.int32)

        # Convert idx to idx tensor for find num of true bbox by img idx
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

        # Get true bbox num array
        ground_truth_num = tf.TensorArray(tf.int32, size=tf.math.reduce_max(idx) + 1, dynamic_size=True, clear_after_read=False)
        for i in tf.range(tf.math.reduce_max(idx) + 1):
            ground_truth_num = ground_truth_num.write(i, tf.zeros(tf.math.reduce_max(count), dtype=tf.int32))

        # sort by bbox confidence score
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
        recalls = tf.math.divide(tp_cumsum, (total_true_bboxes + epsilon))
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


def get_logger(logger_name, log_path):
    logger = logging.getLogger(logger_name)

    # Check handler exists
    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    # info_logger = logging.StreamHandler()
    info_logger = logging.FileHandler(log_path)
    info_logger.setLevel(logging.INFO)
    info_logger.setFormatter(formatter)
    logger.addHandler(info_logger)

    return logger


def get_tagged_img(img, out, prediction=True):
    """
    tagging result on img

    Parameters:
        img (Numpy Array): Image array
        out (Tensor): true label or prediction (1, 7, 7, 30)
        prediction (Bool): true-prediction, false-true label

    Returns:
        Numpy Array: tagged image array
    """
    if not tf.is_tensor(out):
        out = tf.cast(out, dtype=tf.float32)
    # Get all bbox
    all_bboxes = get_all_bboxes(out)

    if prediction:
        bboxes = non_max_suppression(all_bboxes[0])
    else:
        bboxes = tf.gather(all_bboxes[0],
                           tf.reshape(tf.where(all_bboxes[0][..., 1] > 0.4), shape=(-1,)))

    width = img.shape[1]
    height = img.shape[0]
    with open('/home/fssv2/myungsang/datasets/voc_2007/voc.names', 'r') as f:
        class_name_list = f.readlines()
    class_name_list = [x.strip() for x in class_name_list]
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
        img = cv2.putText(img, "{:s}, {:.4f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=(0, 0, 255))

    return img


def get_result_of_yolo_v1_inference(predictions, S=7, B=2, C=20):
    """
    Postprocessing of YOLO V1 predictions

    Parameters:
        predictions (Numpy): predictions of YOLO V1 (batch, S * S * (B*5)+C)

    Returns:
        Numpy: Numpy Array of bboxes (batch, 6(class_index, confidence, x, y, w, h))
    """

    # Reshape of predictions
    predictions = np.reshape(predictions, (-1, S, S, (B*5)+C))

    #############################
    # Get All Bounding Boxes
    #############################
    bbox1_start_index = C + 1
    bbox1_confidence_index = C
    bbox2_start_index = C + 1 + 5
    bbox2_confidence_index = C + 5

    bboxes1 = predictions[..., bbox1_start_index:bbox1_start_index + 4]  # (batch, S, S, 4)
    bboxes2 = predictions[..., bbox2_start_index:bbox2_start_index + 4]  # (batch, S, S, 4)
    confidences = np.concatenate((predictions[..., bbox1_confidence_index:bbox1_confidence_index + 1],
                                  predictions[..., bbox2_confidence_index:bbox2_confidence_index + 1]),
                                 axis=-1)  # (batch, S, S, 2)

    # Get best box index
    best_box_index = np.argmax(confidences, axis=-1)  # (batch, S, S)
    best_box_index = np.expand_dims(best_box_index, axis=-1)  # (batch, S, S, 1)
    best_box_index = best_box_index.astype(np.float32)

    # Get best boxes
    best_boxes = (1 - best_box_index) * bboxes1 + best_box_index * bboxes2  # (batch, S, S, 4)

    # Get cell indexes array
    base_arr = np.arange(S).reshape((1, -1)).repeat(S, axis=0)
    x_cell_indexes = np.expand_dims(base_arr, axis=-1)  # (S, S, 1)

    y_cell_indexes = np.transpose(base_arr)
    y_cell_indexes = np.expand_dims(y_cell_indexes, axis=-1)  # (S, S, 1)

    # Convert x, y ratios to YOLO ratios
    x = 1 / S * (best_boxes[..., :1] + x_cell_indexes)  # (batch, S, S, 1)
    y = 1 / S * (best_boxes[..., 1:2] + y_cell_indexes)  # (batch, S, S, 1)

    # Get class indexes
    class_indexes = np.argmax(predictions[..., :C], axis=-1)  # (batch, S, S)
    class_indexes = np.expand_dims(class_indexes, axis=-1)  # (batch, S, S, 1)
    class_indexes = class_indexes.astype(np.float32)

    # Get best confidences
    best_confidences = (1 - best_box_index) * predictions[..., bbox1_confidence_index:bbox1_confidence_index + 1] + \
                       best_box_index * predictions[..., bbox2_confidence_index:bbox2_confidence_index + 1]  # (batch, S, S, 1)

    # Get converted bboxes
    converted_bboxes = np.concatenate((x, y, best_boxes[..., 2:4]), axis=-1)  # (batch, S, S, 4)

    # Concatenate result
    converted_out = np.concatenate((class_indexes, best_confidences, converted_bboxes), axis=-1)  # (batch, S, S, 6)

    # Get all bboxes
    converted_out = np.reshape(converted_out, (-1, S * S, 6))  # (batch, S*S, 6)

    print(converted_out[1:, :1, :])


if __name__ == "__main__":
    y_true = np.zeros((2, 7, 7, 30))
    y_true[:, 0, 0, 2] = 1  # class
    y_true[:, 0, 0, 20] = 1  # confidence
    y_true[:, 0, 0, 21:25] = (0.5, 0.5, 0.1, 0.1)
    print("y_true:\n{}".format(y_true))

    y_pred = np.zeros((2, 7, 7, 30))
    y_pred[:, 0, 0, 2] = 0.6  # class
    y_pred[:, 0, 0, 20] = 0.7  # confidence
    y_pred[:, 0, 0, 21:25] = (0.49, 0.49, 0.09, 0.09)
    y_pred[:, 0, 0, 25] = 0.4  # confidence
    y_pred[:, 0, 0, 26:30] = (0.45, 0.45, 0.09, 0.09)

    y_pred[:, 1, 0, 2] = 0.6  # class
    y_pred[:, 1, 0, 20] = 0.8  # confidence
    y_pred[:, 1, 0, 21:25] = (0.49, 0.49, 0.09, 0.09)

    y_pred[:, 2, 0, 2] = 0.6  # class
    y_pred[:, 2, 0, 20] = 0.5  # confidence
    y_pred[:, 2, 0, 21:25] = (0.49, 0.49, 0.09, 0.09)

    y_pred[:, 3, 0, 2] = 0.6  # class
    y_pred[:, 3, 0, 20] = 0.9  # confidence
    y_pred[:, 3, 0, 21:25] = (0.49, 0.49, 0.09, 0.09)

    y_pred[:, 4, 0, 2] = 0.6  # class
    y_pred[:, 4, 0, 20] = 0.2  # confidence
    y_pred[:, 4, 0, 21:25] = (0.49, 0.49, 0.09, 0.09)

    y_pred[:, 5, 0, 2] = 0.6  # class
    y_pred[:, 5, 0, 20] = 0.3  # confidence
    y_pred[:, 5, 0, 21:25] = (0.49, 0.49, 0.09, 0.09)
    print("y_pred:\n{}".format(y_pred))

    test_y_pred = np.reshape(y_pred, (2, 1470))
    np_y_true = y_true
    np_y_pred = y_pred

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    iou_tensor = intersection_over_union(y_true[..., 21:25], y_pred[..., 21:25])
    print(iou_tensor)
    a = intersection_over_union_numpy(np_y_true[..., 21:25], np_y_pred[..., 21:25])
    print(a)

    true_bboxes = get_all_bboxes(y_true)
    pred_bboxes = get_all_bboxes(y_pred)
    # print(true_bboxes[1:, :1, :])
    print(pred_bboxes[1:, :1, :])

    pred_bboxes_numpy = get_all_bboxes_numpy(np_y_pred)
    print(pred_bboxes_numpy[1:, :1, :])

    # a = time.time()
    # nms_bboxes = non_max_suppression(pred_bboxes[0], iou_threshold=0.5, threshold=0.4)
    # print('nmx_bboxes: ', nms_bboxes, '\ntaken_time: ', time.time() - a)
    a = time.time()
    nms_bboxes_2 = non_max_suppression_2(pred_bboxes[0], iou_threshold=0.5, threshold=0.4)
    print('nmx_bboxes_2: ', nms_bboxes_2, '\ntaken_time: ', time.time() - a)

    a = time.time()
    nms_bboxes_numpy = non_max_suppression_numpy(pred_bboxes_numpy[0])
    print('nmx_bboxes_numpy: ', nms_bboxes_numpy, '\ntaken_time: ', time.time() - a)

    # all_pred_bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # all_pred_bboxes = all_pred_bboxes.write(all_pred_bboxes.size(), tf.concat([tf.constant([0], dtype=tf.float32), nms_bboxes[0]], axis=0))
    # print(all_pred_bboxes.stack())
    #
    # true_bboxes = non_max_suppression(true_bboxes[0], iou_threshold=0.5, threshold=0.4)
    # print(true_bboxes)
    #
    # all_true_bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # all_true_bboxes = all_true_bboxes.write(all_true_bboxes.size(), tf.concat([tf.constant([0], dtype=tf.float32), true_bboxes[0]], axis=0))
    # print(all_true_bboxes.stack())
    #
    # tmp_true_bboxes = tf.constant([[1., 2., 1., 0.2, 0.5, 0.1, 0.1],
    #                                [1., 2., 1., 0.3, 0.3, 0.1, 0.1],
    #                                [0., 2., 1., 0.3, 0.3, 0.1, 0.1],
    #                                [0., 2., 1., 0.3, 0.3, 0.1, 0.1],
    #                                [0., 2., 1., 0.3, 0.3, 0.1, 0.1],
    #                                [1., 2., 1., 0.4, 0.4, 0.1, 0.1]], dtype=tf.float32)
    #
    # tmp_pred_bboxes = tf.constant([[1., 2., 0.5, 0.2, 0.5, 0.1, 0.1],
    #                                [1., 2., 0.6, 0.3, 0.3, 0.1, 0.1],
    #                                [1., 2., 0.7, 0.8, 0.8, 0.1, 0.1],
    #                                [1., 2., 0.8, 0.8, 0.8, 0.1, 0.1],
    #                                [0., 2., 0.8, 0.8, 0.8, 0.1, 0.1],
    #                                [0., 2., 0.8, 0.8, 0.8, 0.1, 0.1],
    #                                [0., 2., 0.8, 0.8, 0.8, 0.1, 0.1],
    #                                [0., 2., 0.8, 0.8, 0.8, 0.1, 0.1],
    #                                [1., 2., 0.9, 0.8, 0.8, 0.1, 0.1],
    #                                [1., 2., 0.95, 0.4, 0.4, 0.1, 0.1]], dtype=tf.float32)
    #
    # start_time = time.time()
    # map = mean_average_precision(tmp_true_bboxes, tmp_pred_bboxes)
    # print(map)
    # print('Taken_Time: {:.8f}'.format(time.time() - start_time))
    #
    # start_time = time.time()
    # map = mean_average_precision_2(tmp_true_bboxes, tmp_pred_bboxes)
    # print(map)
    # print('Taken_Time: {:.8f}'.format(time.time() - start_time))
