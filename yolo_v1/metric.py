from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import mean_average_precision, get_all_bboxes, non_max_suppression, mean_average_precision_2, non_max_suppression_2
import time


class MeanAveragePrecision:
    def __init__(self):
        self.all_true_bboxes_variable = tf.Variable([[-1, -1, -1, -1, -1, -1, -1]], dtype=tf.float32, shape=tf.TensorShape((None, 7)))
        self.all_pred_bboxes_variable = tf.Variable([[-1, -1, -1, -1, -1, -1, -1]], dtype=tf.float32, shape=tf.TensorShape((None, 7)))
        self.img_idx = tf.Variable([0], dtype=tf.float32)
        self.count = tf.Variable(0, dtype=tf.float32)

    def reset_states(self):
        self.img_idx.assign([0])
        self.count.assign(0)

    def update_state(self, y_true, y_pred):
        true_bboxes = get_all_bboxes(y_true)
        pred_bboxes = get_all_bboxes(y_pred)

        all_true_bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        all_pred_bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for idx in tf.range(tf.shape(y_true)[0]):
            nms_bboxes = non_max_suppression(pred_bboxes[idx], iou_threshold=0.5, threshold=0.4)

            for nms_bbox in nms_bboxes:
                all_pred_bboxes = all_pred_bboxes.write(all_pred_bboxes.size(), tf.concat([self.img_idx, nms_bbox], axis=-1))

            for true_bbox in true_bboxes[idx]:
                if true_bbox[1] > 0.4:
                    all_true_bboxes = all_true_bboxes.write(all_true_bboxes.size(), tf.concat([self.img_idx, true_bbox], axis=-1))

            self.img_idx.assign_add([1])

        if self.count == 0.:
            self.all_true_bboxes_variable.assign(all_true_bboxes.stack())
            self.all_pred_bboxes_variable.assign(all_pred_bboxes.stack())
        else:
            self.all_true_bboxes_variable.assign(tf.concat([self.all_true_bboxes_variable, all_true_bboxes.stack()], axis=0))
            self.all_pred_bboxes_variable.assign(tf.concat([self.all_pred_bboxes_variable, all_pred_bboxes.stack()], axis=0))

        self.count.assign_add(1)

        all_true_bboxes.close()
        all_pred_bboxes.close()
        tf.print('img_idx: ', self.img_idx)
        tf.print('count: ', self.count)

    def result(self):
        return mean_average_precision(self.all_true_bboxes_variable, self.all_pred_bboxes_variable)


class MeanAveragePrecision2:
    def __init__(self):
        self.all_true_bboxes_variable = tf.Variable([[-1, -1, -1, -1, -1, -1, -1]],
                                                    dtype=tf.float32,
                                                    shape=tf.TensorShape((None, 7)))
        self.all_pred_bboxes_variable = tf.Variable([[-1, -1, -1, -1, -1, -1, -1]],
                                                    dtype=tf.float32,
                                                    shape=tf.TensorShape((None, 7)))
        self.img_idx = tf.Variable([0], dtype=tf.float32)

    def reset_states(self):
        self.img_idx.assign([0])

    def update_state(self, y_true, y_pred):
        true_bboxes = get_all_bboxes(y_true)
        pred_bboxes = get_all_bboxes(y_pred)

        for idx in tf.range(tf.shape(y_true)[0]):
            nms_bboxes = non_max_suppression_2(pred_bboxes[idx], iou_threshold=0.5, threshold=0.4)
            pred_img_idx = tf.zeros([tf.shape(nms_bboxes)[0], 1], tf.float32) + self.img_idx
            pred_concat = tf.concat([pred_img_idx, nms_bboxes], axis=1)

            true_bbox = tf.gather(true_bboxes[idx], tf.reshape(tf.where(true_bboxes[idx][..., 1] > 0.4), shape=(-1,)))
            true_img_idx = tf.zeros([tf.shape(true_bbox)[0], 1], tf.float32) + self.img_idx
            true_concat = tf.concat([true_img_idx, true_bbox], axis=1)

            if self.img_idx == 0.:
                self.all_true_bboxes_variable.assign(true_concat)
                self.all_pred_bboxes_variable.assign(pred_concat)
            else:
                self.all_true_bboxes_variable.assign(
                    tf.concat([self.all_true_bboxes_variable, true_concat], axis=0))
                self.all_pred_bboxes_variable.assign(
                    tf.concat([self.all_pred_bboxes_variable, pred_concat], axis=0))

            self.img_idx.assign_add([1])

    def result(self):
        tf.print('all true bboxes: ', tf.shape(self.all_true_bboxes_variable)[0])
        tf.print('all pred bboxes: ', tf.shape(self.all_pred_bboxes_variable)[0])
        return mean_average_precision_2(self.all_true_bboxes_variable, self.all_pred_bboxes_variable)


if __name__ == '__main__':
    tmp_y_true = np.zeros((1, 7, 7, 30))
    tmp_y_true[:, 1, 1, 1] = 1
    tmp_y_true[:, 1, 1, 20] = 1
    tmp_y_true[:, 1, 1, 21:23] = 0.5
    tmp_y_true[:, 1, 1, 23:25] = 0.1
    tmp_y_true[:, 4, 4, 5] = 1
    tmp_y_true[:, 4, 4, 20] = 1
    tmp_y_true[:, 4, 4, 21:23] = 0.5
    tmp_y_true[:, 4, 4, 23:25] = 0.1

    tmp_y_pred = np.zeros((1, 7, 7, 30))
    tmp_y_pred[:, 1, 1, 1] = 0.8
    tmp_y_pred[:, 1, 1, 20] = 0.3
    tmp_y_pred[:, 1, 1, 21:23] = 0.49
    tmp_y_pred[:, 1, 1, 23:25] = 0.1
    tmp_y_pred[:, 1, 1, 25] = 0.2
    tmp_y_pred[:, 1, 1, 26:28] = 0.45
    tmp_y_pred[:, 1, 1, 28:30] = 0.1
    tmp_y_pred[:, 4, 4, 5] = 0.9
    tmp_y_pred[:, 4, 4, 20] = 0.3
    tmp_y_pred[:, 4, 4, 21:23] = 0.49
    tmp_y_pred[:, 4, 4, 23:25] = 0.1
    tmp_y_pred[:, 4, 4, 25] = 0.2
    tmp_y_pred[:, 4, 4, 26:28] = 0.45
    tmp_y_pred[:, 4, 4, 28:30] = 0.1

    tmp_y_true = tf.cast(tmp_y_true, dtype=tf.float32)
    tmp_y_pred = tf.cast(tmp_y_pred, dtype=tf.float32)

    tmp_y_pred_2 = np.zeros((1, 7, 7, 30))
    tmp_y_pred_2[:, 1, 1, 1] = 0.8
    tmp_y_pred_2[:, 1, 1, 20] = 0.9
    tmp_y_pred_2[:, 1, 1, 21:23] = 0.49
    tmp_y_pred_2[:, 1, 1, 23:25] = 0.1
    tmp_y_pred_2[:, 1, 1, 25] = 0.2
    tmp_y_pred_2[:, 1, 1, 26:28] = 0.45
    tmp_y_pred_2[:, 1, 1, 28:30] = 0.1
    tmp_y_pred_2 = tf.cast(tmp_y_pred_2, dtype=tf.float32)

    # map_metric = MeanAveragePrecision()
    map_metric = MeanAveragePrecision2()

    import time
    for _ in range(1):
        for i in range(5):
            if i == 1:
                tmp_y_pred =tmp_y_pred_2
            start_time = time.time()
            map_metric.update_state(tmp_y_true, tmp_y_pred)
            print('taken_time: {:.4f}'.format(time.time() - start_time))
        map = map_metric.result()
        print(map)
        map_metric.reset_states()
