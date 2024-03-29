





# @tf.function
# def non_max_suppression(bboxes, iou_threshold=0.5, threshold=0.4):
#     """
#     Does Non Max Suppression given bboxes

#     Arguments:
#         bboxes (Tensor): tensor of all bboxes with each grid (S*S, 6)
#         specified as [class_idx, confidence_score, x, y, w, h]
#         iou_threshold (float): threshold where predicted bboxes is correct
#         threshold (float): threshold to remove predicted bboxes

#     Returns:
#         Tensor: bboxes after performing NMS given a specific IoU threshold (None, 6)
#     """
#     assert tf.is_tensor(bboxes)

#     # bboxes smaller than the threshold are removed
#     # bboxes = [box for box in bboxes if box[1] > threshold]
#     bboxes_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
#     for bbox in bboxes:
#         if bbox[1] > threshold:
#             bboxes_array = bboxes_array.write(bboxes_array.size(), bbox)

#     # sort descending by confidence score
#     # bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
#     bboxes_array = bboxes_array.scatter(tf.argsort(bboxes_array.stack()[:, 1], direction='DESCENDING'),
#                                         bboxes_array.stack())

#     # get bboxes after nms
#     # bboxes_after_nms = []
#     bboxes_after_nms = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

#     # loop variables
#     loop_vars = (bboxes_array, bboxes_after_nms)

#     # loop condition
#     loop_cond = lambda i, j: not(tf.less(i.size(), 1))

#     # loop body
#     def loop_body(i, j):
#         chosen_box = i.read(0)
#         tmp_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
#         for idx in tf.range(1, i.size()):
#             bbox = i.read(idx)
#             if bbox[0] != chosen_box[0] or intersection_over_union(
#                     chosen_box[2:], bbox[2:]
#             ) < iou_threshold:
#                 tmp_array = tmp_array.write(tmp_array.size(), bbox)
#         i = tmp_array

#         j = j.write(j.size(), chosen_box)

#         return i, j

#     _, bboxes_after_nms = tf.while_loop(loop_cond, loop_body, loop_vars)

#     # # while bboxes:
#     # while bboxes_array.size():
#     #     # chosen_box = bboxes.pop(0)
#     #     chosen_box = bboxes_array.read(0)
#     #
#     #     # bboxes = [
#     #     #     box
#     #     #     for box in bboxes
#     #     #     if box[0] != chosen_box[0]
#     #     #     or intersection_over_union(
#     #     #         tf.cast(chosen_box[2:], dtype=np.float32),
#     #     #         tf.cast(box[2:], dtype=np.float32))
#     #     #        < iou_threshold
#     #     # ]
#     #     tmp_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
#     #     for idx in range(1, bboxes_array.size()):
#     #         bbox = tmp_array.read(idx)
#     #         if bbox[0] != chosen_box[0] or intersection_over_union(
#     #             chosen_box[2:], bbox[2:]
#     #         ) < iou_threshold:
#     #             tmp_array = tmp_array.write(tmp_array.size(), bbox)
#     #     bboxes_array = tmp_array
#     #
#     #     # bboxes_after_nms.append(chosen_box)
#     #     bboxes_after_nms = bboxes_after_nms.write(bboxes_after_nms.size(), chosen_box)

#     return bboxes_after_nms.stack()






# @tf.function
# def get_all_bboxes(out, grid=7, num_classes=20):
#     """Converts bounding boxes output from Yolo with
#        an image split size of S into entire image ratios
#        rather than relative to cell ratios.

#     Arguments:
#         out (Tensor): predictions or true_labels (batch, S, S, (B*5)+C)

#     Returns:
#         Tensor: Tensor of bboxes (batch, S*S, 6(class_index, confidence, x, y, w, h))
#     """
#     assert tf.is_tensor(out)

#     bbox1_start_index = num_classes + 1
#     bbox1_confidence_index = num_classes
#     bbox2_start_index = num_classes + 1 + 5
#     bbox2_confidence_index = num_classes + 5

#     bboxes1 = out[..., bbox1_start_index:bbox1_start_index+4]  # (batch, S, S, 4)
#     bboxes2 = out[..., bbox2_start_index:bbox2_start_index+4] # (batch, S, S, 4)
#     confidences = tf.concat([out[..., bbox1_confidence_index:bbox1_confidence_index+1],
#                              out[..., bbox2_confidence_index:bbox2_confidence_index+1]], axis=-1) # (batch, S, S, 2)

#     # Get best box index
#     best_box_index = tf.math.argmax(confidences, axis=-1) # (batch, S, S)
#     best_box_index = tf.expand_dims(best_box_index, axis=-1) # (batch, S, S, 1)
#     best_box_index = tf.cast(best_box_index, dtype=np.float32)

#     # Get best boxes
#     best_boxes = (1 - best_box_index) * bboxes1 + best_box_index * bboxes2 # (batch, S, S, 4)

#     # Get cell indexes array
#     base_arr = tf.map_fn(fn=lambda x: tf.range(x, x + grid), elems=tf.zeros(grid))
#     x_cell_indexes = tf.reshape(base_arr, shape=(grid, grid, 1)) # (S, S, 1)

#     y_cell_indexes = tf.transpose(base_arr)
#     y_cell_indexes = tf.reshape(y_cell_indexes, shape=(grid, grid, 1)) # (S, S, 1)

#     # Convert x, y ratios to YOLO ratios
#     x = 1 / grid * (best_boxes[..., :1] + x_cell_indexes) # (batch, S, S, 1)
#     y = 1 / grid * (best_boxes[..., 1:2] + y_cell_indexes) # (batch, S, S, 1)

#     # Get class indexes
#     class_indexes = tf.math.argmax(out[..., :num_classes], axis=-1) # (batch, S, S)
#     class_indexes = tf.expand_dims(class_indexes, axis=-1) # (batch, S, S, 1)
#     class_indexes = tf.cast(class_indexes, dtype=np.float32)

#     # Get best confidences
#     best_confidences = (1 - best_box_index) * out[..., bbox1_confidence_index:bbox1_confidence_index+1] + \
#                        best_box_index * out[..., bbox2_confidence_index:bbox2_confidence_index+1] # (batch, S, S, 1)

#     # Get converted bboxes
#     converted_bboxes = tf.concat([x, y, best_boxes[..., 2:4]], axis=-1) # (batch, S, S, 4)

#     # Concatenate result
#     converted_out = tf.concat([class_indexes, best_confidences, converted_bboxes], axis=-1) # (batch, S, S, 6)

#     # Get all bboxes
#     converted_out = tf.reshape(converted_out, shape=(-1, grid * grid, 6)) # (batch, S*S, 6)

#     return converted_out # (batch, S*S, 6)




# def get_logger(logger_name, log_path):
#     logger = logging.getLogger(logger_name)

#     # Check handler exists
#     if len(logger.handlers) > 0:
#         return logger

#     logger.setLevel(logging.INFO)

#     formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')

#     # info_logger = logging.StreamHandler()
#     info_logger = logging.FileHandler(log_path)
#     info_logger.setLevel(logging.INFO)
#     info_logger.setFormatter(formatter)
#     logger.addHandler(info_logger)

#     return logger







# @tf.function(input_signature=[tf.TensorSpec(shape=[None, 7]), tf.TensorSpec(shape=[None, 7])])
# def mean_average_precision(true_bboxes, pred_bboxes, iou_threshold=0.5, num_classes=20):
#     print('mean average precision tracing')
#     """
#     Calculates mean average precision

#     Arguments:
#         true_bboxes (Tensor): Tensor of all bboxes with all images (None, 7)
#         specified as [img_idx, class_idx, confidence_score, x, y, w, h]
#         pred_bboxes (Tensor): Similar as true_bboxes
#         iou_threshold (float): threshold where predicted bboxes is correct
#         num_classes (int): number of classes

#     Returns:
#         float: mAP value across all classes given a specific IoU threshold
#     """
#     assert tf.is_tensor(true_bboxes) and tf.is_tensor(pred_bboxes)

#     # list storing all AP for respective classes
#     average_precisions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

#     # used for numerical stability later on
#     epsilon = 1e-6

#     for c in tf.range(num_classes, dtype=tf.float32):
#         tf.print('calculating AP: ', c, ' / ', num_classes)
#         # detections, ground_truths variables in specific class
#         detections = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
#         ground_truths = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

#         # Go through all predictions and targets,
#         # and only add the ones that belong to the
#         # current class c
#         for detection in pred_bboxes:
#             if detection[1] == c:
#                 detections = detections.write(detections.size(), detection)

#         for true_box in true_bboxes:
#             if true_box[1] == c:
#                 ground_truths = ground_truths.write(ground_truths.size(), true_box)

#         # If none exists for this class then we can safely skip
#         # total_true_bboxes = len(ground_truths)
#         total_true_bboxes = tf.cast(ground_truths.size(), dtype=tf.float32)
#         if total_true_bboxes == 0.:
#             average_precisions = average_precisions.write(average_precisions.size(),
#                                                           tf.constant(0, dtype=tf.float32))
#             continue

#         # tf.print(c, ' class ground truths size: ', ground_truths.size())
#         # tf.print(c, ' class detections size: ', detections.size())

#         # Get the number of true bbox by image index
#         img_idx, idx, count = tf.unique_with_counts(ground_truths.stack()[..., 0])
#         img_idx = tf.cast(img_idx, dtype=tf.int32)

#         # Convert idx to idx tensor for find num of true bbox by img idx
#         idx_tensor = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
#         for i in tf.range(tf.math.reduce_max(idx) + 1):
#             idx_tensor = idx_tensor.write(idx_tensor.size(), i)
#         idx_tensor = idx_tensor.stack()

#         # Get hash table: key - img_idx, value - idx_tensor
#         table = tf.lookup.experimental.DenseHashTable(
#             key_dtype=tf.int32,
#             value_dtype=tf.int32,
#             default_value=-1,
#             empty_key=-1,
#             deleted_key=-2
#         )
#         table.insert(img_idx, idx_tensor)
#         # tf.print('table: ', table.export())
#         # tf.print('table[0]: ', table.lookup(tf.constant([0], dtype=tf.int32)))

#         # Get true bbox num array
#         ground_truth_num = tf.TensorArray(tf.int32, size=tf.math.reduce_max(idx) + 1, dynamic_size=True, clear_after_read=False)
#         for i in tf.range(tf.math.reduce_max(idx) + 1):
#             ground_truth_num = ground_truth_num.write(i, tf.zeros(tf.math.reduce_max(count), dtype=tf.int32))

#         # find the amount of bboxes for each training example
#         # Counter here finds how many ground truth bboxes we get
#         # for each training example, so let's say img 0 has 3,
#         # img 1 has 5 then we will obtain a dictionary with:
#         # amount_bboxes = {0:3, 1:5}
#         # amount_bboxes = Counter([gt[0] for gt in ground_truths])

#         # We then go through each key, val in this dictionary
#         # and convert to the following (w.r.t same example):
#         # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
#         # for key, val in amount_bboxes.items():
#         #     amount_bboxes[key] = np.zeros(val)

#         # sort by bbox confidence score
#         # detections.sort(key=lambda x: x[2], reverse=True)
#         # TP = np.zeros((len(detections)))
#         # FP = np.zeros((len(detections)))
#         detections = detections.gather(tf.argsort(detections.stack()[..., 2], direction='DESCENDING'))
#         true_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())
#         false_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())

#         detections_size = tf.shape(detections)[0]
#         detection_idx = tf.constant(0, dtype=tf.int32)
#         for detection in detections:
#             # tf.print('progressing of detection: ', detection_idx, ' / ', detections_size)
#             # Only take out the ground_truths that have the same
#             # training idx as detection
#             # ground_truth_img = [
#             #     bbox for bbox in ground_truths if bbox[0] == detection[0]
#             # ]
#             # tf.print('detection_img_idx: ', detection[0])
#             # tf.print('detection_confidence: ', detection[2])

#             ground_truth_img = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
#             for ground_truth_idx in tf.range(ground_truths.size()):
#                 if ground_truths.read(ground_truth_idx)[0] == detection[0]:
#                     ground_truth_img = ground_truth_img.write(ground_truth_img.size(), ground_truths.read(ground_truth_idx))
#             # tf.print('ground_truth_img: ', ground_truth_img.size())

#             best_iou = tf.TensorArray(tf.float32, size=1, element_shape=(1,), clear_after_read=False)
#             best_gt_idx = tf.TensorArray(tf.int32, size=1, element_shape=(), clear_after_read=False)

#             for idx in tf.range(ground_truth_img.size()):
#                 # iou = intersection_over_union(
#                 #     tf.cast(detection[3:], dtype=tf.float32),
#                 #     tf.cast(gt[3:], dtype=tf.float32)
#                 # )

#                 iou = intersection_over_union(
#                     detection[3:],
#                     ground_truth_img.read(idx)[3:]
#                 )

#                 if iou > best_iou.read(0):
#                     best_iou = best_iou.write(0, iou)
#                     best_gt_idx = best_gt_idx.write(0, idx)

#             if best_iou.read(0) > iou_threshold:
#                 # # only detect ground truth detection once
#                 # if amount_bboxes[detection[0]][best_gt_idx] == 0:
#                 #     # true positive and add this bounding box to seen
#                 #     TP[detection_idx] = 1
#                 #     amount_bboxes[detection[0]][best_gt_idx] = 1
#                 # else:
#                 #     FP[detection_idx] = 1

#                 # Get current detections img_idx
#                 cur_det_img_idx = tf.cast(detection[0], dtype=tf.int32)

#                 # Get row idx of ground_truth_num array
#                 gt_row_idx = table.lookup(cur_det_img_idx)

#                 # Get 'current img ground_truth_num tensor'
#                 cur_gt_num_tensor = ground_truth_num.read(gt_row_idx)

#                 # Get idx of current best ground truth
#                 cur_best_gt_idx = best_gt_idx.read(0)

#                 if cur_gt_num_tensor[cur_best_gt_idx] == 0:
#                     true_positive = true_positive.write(detection_idx, 1)

#                     # change cur_gt_num_tensor[cur_best_gt_idx] to 1
#                     cur_gt_num_tensor = change_tensor(cur_gt_num_tensor, cur_best_gt_idx)

#                     # update ground_truth_num array
#                     ground_truth_num = ground_truth_num.write(gt_row_idx, cur_gt_num_tensor)

#                 else:
#                     false_positive = false_positive.write(detection_idx, 1)

#             # if IOU is lower then the detection is a false positive
#             else:
#                 false_positive = false_positive.write(detection_idx, 1)

#             ground_truth_img.close()
#             best_iou.close()
#             best_gt_idx.close()
#             detection_idx += 1

#         # Compute the cumulative sum of the tensor
#         # tf.print("TP: ", true_positive.stack())
#         # tf.print("FP: ", false_positive.stack())
#         tp_cumsum = tf.math.cumsum(true_positive.stack(), axis=0)
#         fp_cumsum = tf.math.cumsum(false_positive.stack(), axis=0)

#         # Calculate recalls and precisions
#         recalls = tf.math.divide(tp_cumsum, (total_true_bboxes + epsilon))
#         precisions = tf.math.divide(tp_cumsum, (tp_cumsum + fp_cumsum + epsilon))

#         # Append start point value of precision-recall graph
#         precisions = tf.concat([tf.constant([1], dtype=tf.float32), precisions], axis=0)
#         recalls = tf.concat([tf.constant([0], dtype=tf.float32), recalls], axis=0)
#         # tf.print(precisions)
#         # tf.print(recalls)

#         # Calculate area of precision-recall graph
#         average_precision_value = tf.py_function(func=np.trapz,
#                                                  inp=[precisions, recalls],
#                                                  Tout=tf.float32)
#         average_precisions = average_precisions.write(average_precisions.size(), average_precision_value)
#         # tf.print('average precision: ', average_precision_value)

#         ground_truths.close()
#         ground_truth_num.close()
#         true_positive.close()
#         false_positive.close()

#         # TP_cumsum = np.cumsum(TP, axis=0)
#         # FP_cumsum = np.cumsum(FP, axis=0)
#         # recalls = TP_cumsum / (total_true_bboxes + epsilon)
#         # precisions = np.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
#         # precisions = np.concatenate(([1.], precisions), axis=0)
#         # recalls = np.concatenate(([0.], recalls), axis=0)
#         # # tf.math.trapz for numerical integration
#         # average_precisions.append(np.trapz(precisions, recalls))
#     # tf.print(average_precisions.stack())
#     tf.print('mAP: ', tf.math.reduce_mean(average_precisions.stack()))
#     return tf.math.reduce_mean(average_precisions.stack())







# @tf.function
# def change_tensor(tensor_1d, idx_col):
#     """
#     change the value of a specific column in a tensor to 1

#     Arguments:
#         tensor_1d (Tensor): 1D Tensor to change
#         idx_col (Tensor): index of specific column to change

#     Returns:
#         Tensor: changed tensor_1d
#     """
#     dst = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
#     idx = tf.constant(0)
#     for value in tensor_1d:
#         if idx == idx_col:
#             dst = dst.write(dst.size(), tf.constant(1))
#         else:
#             dst = dst.write(dst.size(), value)
#         idx += 1
#     return dst.stack()








# @tf.function(input_signature=[tf.TensorSpec(shape=[None, 7]), tf.TensorSpec(shape=[None, 7])])
# def mean_average_precision_2(true_bboxes, pred_bboxes, iou_threshold=0.5, num_classes=20):
#     print('mean average precision tracing')
#     """
#     Calculates mean average precision

#     Arguments:
#         true_bboxes (Tensor): Tensor of all bboxes with all images (None, 7)
#         specified as [img_idx, class_idx, confidence_score, x, y, w, h]
#         pred_bboxes (Tensor): Similar as true_bboxes
#         iou_threshold (float): threshold where predicted bboxes is correct
#         num_classes (int): number of classes

#     Returns:
#         float: mAP value across all classes given a specific IoU threshold
#     """
#     assert tf.is_tensor(true_bboxes) and tf.is_tensor(pred_bboxes)

#     # list storing all AP for respective classes
#     average_precisions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

#     # used for numerical stability later on
#     epsilon = 1e-6

#     for c in tf.range(num_classes, dtype=tf.float32):
#         tf.print('calculating AP: ', c, ' / ', num_classes)
#         # detections, ground_truths variables in specific class
#         detections = tf.gather(pred_bboxes, tf.reshape(tf.where(pred_bboxes[..., 1] == c), shape=(-1,)))
#         ground_truths = tf.gather(true_bboxes, tf.reshape(tf.where(true_bboxes[..., 1] == c), shape=(-1,)))

#         # If none exists for this class then we can safely skip
#         total_true_bboxes = tf.cast(tf.shape(ground_truths)[0], dtype=tf.float32)
#         if total_true_bboxes == 0.:
#             average_precisions = average_precisions.write(average_precisions.size(),
#                                                           tf.constant(0, dtype=tf.float32))
#             continue

#         # tf.print(c, ' class ground truths size: ', tf.shape(ground_truths)[0])
#         # tf.print(c, ' class detections size: ', tf.shape(detections)[0])

#         # Get the number of true bbox by image index
#         img_idx, idx, count = tf.unique_with_counts(ground_truths[..., 0])
#         img_idx = tf.cast(img_idx, dtype=tf.int32)

#         # Convert idx to idx tensor for find num of true bbox by img idx
#         idx_tensor = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
#         for i in tf.range(tf.math.reduce_max(idx) + 1):
#             idx_tensor = idx_tensor.write(idx_tensor.size(), i)
#         idx_tensor = idx_tensor.stack()

#         # Get hash table: key - img_idx, value - idx_tensor
#         table = tf.lookup.experimental.DenseHashTable(
#             key_dtype=tf.int32,
#             value_dtype=tf.int32,
#             default_value=-1,
#             empty_key=-1,
#             deleted_key=-2
#         )
#         table.insert(img_idx, idx_tensor)

#         # Get true bbox num array
#         ground_truth_num = tf.TensorArray(tf.int32, size=tf.math.reduce_max(idx) + 1, dynamic_size=True, clear_after_read=False)
#         for i in tf.range(tf.math.reduce_max(idx) + 1):
#             ground_truth_num = ground_truth_num.write(i, tf.zeros(tf.math.reduce_max(count), dtype=tf.int32))

#         # sort by bbox confidence score
#         detections = tf.gather(detections, tf.argsort(detections[..., 2], direction='DESCENDING'))
#         true_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())
#         false_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())

#         detections_size = tf.shape(detections)[0]
#         detection_idx = tf.constant(0, dtype=tf.int32)
#         for detection in detections:
#             # tf.print('progressing of detection: ', detection_idx, ' / ', detections_size)
#             # tf.print('detection_img_idx: ', detection[0])
#             # tf.print('detection_confidence: ', detection[2])

#             ground_truth_img = tf.gather(ground_truths, tf.reshape(tf.where(ground_truths[..., 0] == detection[0]),
#                                                                    shape=(-1,)))
#             # tf.print('ground_truth_img: ', tf.shape(ground_truth_img)[0])

#             best_iou = tf.TensorArray(tf.float32, size=1, element_shape=(1,), clear_after_read=False)
#             best_gt_idx = tf.TensorArray(tf.int32, size=1, element_shape=(), clear_after_read=False)

#             gt_idx = tf.constant(0, dtype=tf.int32)
#             for gt_img in ground_truth_img:
#                 iou = intersection_over_union(detection[3:], gt_img[3:])

#                 if iou > best_iou.read(0):
#                     best_iou = best_iou.write(0, iou)
#                     best_gt_idx = best_gt_idx.write(0, gt_idx)

#                 gt_idx += 1

#             if best_iou.read(0) > iou_threshold:
#                 # Get current detections img_idx
#                 cur_det_img_idx = tf.cast(detection[0], dtype=tf.int32)

#                 # Get row idx of ground_truth_num array
#                 gt_row_idx = table.lookup(cur_det_img_idx)

#                 # Get 'current img ground_truth_num tensor'
#                 cur_gt_num_tensor = ground_truth_num.read(gt_row_idx)

#                 # Get idx of current best ground truth
#                 cur_best_gt_idx = best_gt_idx.read(0)

#                 if cur_gt_num_tensor[cur_best_gt_idx] == 0:
#                     true_positive = true_positive.write(detection_idx, 1)

#                     # change cur_gt_num_tensor[cur_best_gt_idx] to 1
#                     cur_gt_num_tensor = change_tensor(cur_gt_num_tensor, cur_best_gt_idx)

#                     # update ground_truth_num array
#                     ground_truth_num = ground_truth_num.write(gt_row_idx, cur_gt_num_tensor)

#                 else:
#                     false_positive = false_positive.write(detection_idx, 1)

#             # if IOU is lower then the detection is a false positive
#             else:
#                 false_positive = false_positive.write(detection_idx, 1)

#             # ground_truth_img.close()
#             best_iou.close()
#             best_gt_idx.close()
#             detection_idx += 1

#         # Compute the cumulative sum of the tensor
#         tp_cumsum = tf.math.cumsum(true_positive.stack(), axis=0)
#         fp_cumsum = tf.math.cumsum(false_positive.stack(), axis=0)

#         # Calculate recalls and precisions
#         recalls = tf.math.divide(tp_cumsum, (total_true_bboxes + epsilon))
#         precisions = tf.math.divide(tp_cumsum, (tp_cumsum + fp_cumsum + epsilon))

#         # Append start point value of precision-recall graph
#         precisions = tf.concat([tf.constant([1], dtype=tf.float32), precisions], axis=0)
#         recalls = tf.concat([tf.constant([0], dtype=tf.float32), recalls], axis=0)
#         # tf.print(precisions)
#         # tf.print(recalls)

#         # Calculate area of precision-recall graph
#         average_precision_value = tf.py_function(func=np.trapz,
#                                                  inp=[precisions, recalls],
#                                                  Tout=tf.float32)
#         average_precisions = average_precisions.write(average_precisions.size(), average_precision_value)
#         # tf.print('average precision: ', average_precision_value)

#         ground_truth_num.close()
#         true_positive.close()
#         false_positive.close()

#     # tf.print(average_precisions.stack())
#     tf.print('mAP: ', tf.math.reduce_mean(average_precisions.stack()))
#     return tf.math.reduce_mean(average_precisions.stack())











# class MeanAveragePrecision:
#     def __init__(self):
#         self.all_true_bboxes_variable = tf.Variable([[-1, -1, -1, -1, -1, -1, -1]],
#                                                     dtype=tf.float32,
#                                                     shape=tf.TensorShape((None, 7)))
#         self.all_pred_bboxes_variable = tf.Variable([[-1, -1, -1, -1, -1, -1, -1]],
#                                                     dtype=tf.float32,
#                                                     shape=tf.TensorShape((None, 7)))
#         self.img_idx = tf.Variable([0], dtype=tf.float32)

#     def reset_states(self):
#         self.img_idx.assign([0])

#     def update_state(self, y_true, y_pred):
#         true_bboxes = get_all_bboxes(y_true)
#         pred_bboxes = get_all_bboxes(y_pred)

#         for idx in tf.range(tf.shape(y_true)[0]):
#             nms_bboxes = non_max_suppression(pred_bboxes[idx], iou_threshold=0.5, conf_threshold=0.4)
#             pred_img_idx = tf.zeros([tf.shape(nms_bboxes)[0], 1], tf.float32) + self.img_idx
#             pred_concat = tf.concat([pred_img_idx, nms_bboxes], axis=1)

#             true_bbox = tf.gather(true_bboxes[idx], tf.reshape(tf.where(true_bboxes[idx][..., 1] > 0.4), shape=(-1,)))
#             true_img_idx = tf.zeros([tf.shape(true_bbox)[0], 1], tf.float32) + self.img_idx
#             true_concat = tf.concat([true_img_idx, true_bbox], axis=1)

#             if self.img_idx == 0.:
#                 self.all_true_bboxes_variable.assign(true_concat)
#                 self.all_pred_bboxes_variable.assign(pred_concat)
#             else:
#                 self.all_true_bboxes_variable.assign(
#                     tf.concat([self.all_true_bboxes_variable, true_concat], axis=0))
#                 self.all_pred_bboxes_variable.assign(
#                     tf.concat([self.all_pred_bboxes_variable, pred_concat], axis=0))

#             self.img_idx.assign_add([1])

#     def result(self):
#         tf.print('all true bboxes: ', tf.shape(self.all_true_bboxes_variable)[0])
#         tf.print('all pred bboxes: ', tf.shape(self.all_pred_bboxes_variable)[0])
#         return mean_average_precision_2(self.all_true_bboxes_variable, self.all_pred_bboxes_variable)









# class YoloV1Generator(keras.utils.Sequence):
#     def __init__(self, data_dir, input_shape, batch_size, num_classes, num_boxes, drop_remainder=False, grid=7, augment=False, shuffle=False):
#         self.img_path_array = np.array(glob(data_dir + '/*.jpg'))
#         self.input_shape = input_shape
#         self.output_shape = (grid, grid, num_classes + (num_boxes*5))
#         self.batch_size = batch_size
#         self.drop_remainder = drop_remainder
#         self.grid = grid
#         self.num_boxes = num_boxes
#         self.num_classes = num_classes
#         self.augment = augment
#         self.shuffle = shuffle
#         self.indexes = None
#         self.on_epoch_end()

#     def on_epoch_end(self):
#         self.indexes = np.arange(len(self.img_path_array))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __len__(self):
#         if self.drop_remainder:
#             return int(len(self.img_path_array) // self.batch_size)
#         else:
#             share = float(len(self.img_path_array) // self.batch_size)
#             division_result = float(len(self.img_path_array) / self.batch_size)
#             if division_result - share > 0.:
#                 return int(share + 1)
#             else:
#                 return int(share)

#     def __getitem__(self, index):
#         if self.drop_remainder:
#             indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
#         else:
#             if (index + 1) * self.batch_size >= len(self.img_path_array):
#                 indexes = self.indexes[index * self.batch_size:]
#             else:
#                 indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
#         img_path_array = self.img_path_array[indexes]
#         x, y = self._data_gen(img_path_array)
#         return x, y

#     def _data_gen(self, img_path_array):
#         cv2.setNumThreads(0)
#         if not self.augment:
#             batch_images = np.zeros(
#                 shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
#                 dtype=np.float32
#             )

#             batch_labels = np.zeros(
#                 shape=(self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]),
#                 dtype=np.float32
#             )

#             for i, img_path in enumerate(img_path_array):
#                 image = cv2.imread(img_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
#                 image = image / 255.

#                 label_path = img_path.replace('.jpg', '.txt')
#                 label = self.__get_label_matrix(label_path)

#                 batch_images[i] = image
#                 batch_labels[i] = label

#         else:
#             batch_images = np.zeros(
#                 shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
#                 dtype=np.uint8
#             )

#             batch_labels = np.zeros(
#                 shape=(self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]),
#                 dtype=np.float32
#             )

#             bbs = []

#             for i, img_path in enumerate(img_path_array):
#                 image = cv2.imread(img_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image_resized = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
#                 batch_images[i] = image_resized

#                 label_path = img_path.replace('.jpg', '.txt')
#                 ia_bbox_list = self.__get_ia_bbox_list(label_path)
#                 bbs.append(ia_bbox_list)

#             images_aug, bbs_aug = self.__augment_images_and_bboxes(batch_images, bbs)

#             for i, ia_bboxes in enumerate(bbs_aug):
#                 batch_labels[i] = self._get_label_matrix_from_ia_bbox(ia_bboxes)

#             batch_images = images_aug.astype(np.float32) / 255.

#         return batch_images, batch_labels

#     def _get_label_matrix_from_ia_bbox(self, ia_bboxes):
#         label_matrix = np.zeros(self.output_shape)

#         for ia_bbox in ia_bboxes:
#             # Get Class index, bbox info
#             cls = ia_bbox.label
#             xmin = ia_bbox.x1
#             ymin = ia_bbox.y1
#             xmax = ia_bbox.x2
#             ymax = ia_bbox.y2

#             x = ((xmin + xmax) / 2.) / self.input_shape[0]
#             if x >= 1.: continue
#             y = ((ymin + ymax) / 2.) / self.input_shape[1]
#             if y >= 1.: continue
#             w = (xmax - xmin) / self.input_shape[0]
#             h = (ymax - ymin) / self.input_shape[1]

#             # Start from grid position and calculate x, y
#             loc = [self.grid * y, self.grid * x]
#             loc_i = int(loc[0])
#             loc_j = int(loc[1])
#             y = loc[0] - loc_i
#             x = loc[1] - loc_j

#             if label_matrix[loc_i, loc_j, self.num_classes] == 0: # confidence
#                 label_matrix[loc_i, loc_j, cls] = 1 # class
#                 label_matrix[loc_i, loc_j, self.num_classes+1:self.num_classes+5] = [x, y, w, h]
#                 label_matrix[loc_i, loc_j, self.num_classes] = 1 # confidence

#         return label_matrix

#     def __get_ia_bbox_list(self, label_path):
#         dst = []

#         # Get label data
#         with open(label_path, 'r') as label_file:
#             label_data = label_file.readlines()
#         label_data_list = [y.split(' ') for y in [x.strip() for x in label_data]]

#         for data in label_data_list:
#             class_idx = int(data[0])
#             x = float(data[1]) * self.input_shape[0]
#             y = float(data[2]) * self.input_shape[1]
#             w = float(data[3]) * self.input_shape[0]
#             h = float(data[4]) * self.input_shape[1]

#             xmin = x - (w / 2.)
#             ymin = y - (h / 2.)
#             xmax = x + (w / 2.)
#             ymax = y + (h / 2.)

#             dst.append(ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=class_idx))

#         return dst

#     def __augment_images_and_bboxes(self, images, bboxes):
#         seq = iaa.Sequential(
#             [
#                 iaa.SomeOf((0, 7),  # Apply 1 to <max> given augmenters
#                            # iaa.SomeOf(1,  # Apply 1 of given augmenters
#                            [
#                                iaa.Identity(),  # no change

#                                # imgaug.augmenters.convolutional
#                                iaa.Sharpen(),

#                                # imgaug.augmenters.flip
#                                iaa.Fliplr(), # Vertical
#                                iaa.Flipud(0.5), # Horizontal

#                                # imgaug.aumenters.color
#                                iaa.MultiplyBrightness((0.5, 1.5)),
#                                iaa.MultiplySaturation((0.5, 1.5)),
#                                iaa.MultiplyHue((0.5, 1.5))

#                                # imgaug.augmenters.geometric
#                                # iaa.ScaleX((1.0, 1.2)),
#                                # iaa.ScaleY((1.0, 1.2)),
#                                # iaa.TranslateX(percent=(-0.2, 0.2)),
#                                # iaa.TranslateY(percent=(-0.2, 0.2)),

#                                # iaa.Solarize(threshold=0),  # inverts all pixel values above a threshold
#                                # iaa.HistogramEqualization(),
#                                # iaa.Posterize(nb_bits=(1, 8)),
#                                # iaa.GammaContrast(gamma=(0.5, 2.0)),
#                                # iaa.Rot90(k=(2, 3)),
#                            ]
#                            )
#             ]
#         )
#         return seq(images=images, bounding_boxes=bboxes)

#     def __get_label_matrix(self, label_path):
#         # label matrix = S*S*(B*5 + C)
#         label_matrix = np.zeros(self.output_shape)

#         # Get label data
#         with open(label_path, 'r') as f:
#             label_data = f.readlines()
#         label_data = [data.strip() for data in label_data]

#         for data in label_data:
#             # Get data list of label, bbox info
#             data_list = data.split(' ')
#             data_list = [float(data) for data in data_list]

#             # Get Class index, bbox info
#             cls = int(data_list[0])
#             x = data_list[1]
#             y = data_list[2]
#             w = data_list[3]
#             h = data_list[4]

#             # Start from grid position and calculate x, y
#             loc = [self.grid * y, self.grid * x]
#             loc_i = int(loc[0])
#             loc_j = int(loc[1])
#             y = loc[0] - loc_i
#             x = loc[1] - loc_j

#             if label_matrix[loc_i, loc_j, self.num_classes] == 0: # confidence
#                 label_matrix[loc_i, loc_j, cls] = 1 # class
#                 label_matrix[loc_i, loc_j, self.num_classes+1:self.num_classes+5] = [x, y, w, h]
#                 label_matrix[loc_i, loc_j, self.num_classes] = 1 # confidence

#         return label_matrix







