import os
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import albumentations as A

from utils import MeanAveragePrecisionNumpy, MeanAveragePrecision

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
@tf.function
def intersection_over_union(boxes1, boxes2):
    """Calculation of intersection-over-union

    Arguments:
        boxes1 (Tensor): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [x, y, w, h]
        boxes2 (Tensor): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [x, y, w, h]

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

@tf.function
def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.4):
    """Does Non Max Suppression given bboxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, x, y, w, h]
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

@tf.function
def decode_predictions(predictions, num_classes=20, num_boxes=2):
    """decodes predictions of the YOLO v1 model
    
    Arguments:
        predictions (Tensor): predictions of the YOLO v1 model with shape  '(1, 7, 7, (num_boxes*5 + num_classes))'
        num_classes: Number of classes in the dataset
        num_boxes: Number of boxes to predict

    Returns:
        Tensor: boxes after performing NMS given a specific IoU threshold '(None, 6)'
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
    pred_box = tf.math.reduce_sum(pred_box.stack(), axis=0)
    pred_conf = tf.math.reduce_sum(pred_conf.stack(), axis=0)

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
    
    return non_max_suppression(pred_result[0])


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
    for bbox in boxes:
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

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
        img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=(0, 255, 0))

    return img

##################################
# Dataset Generator
##################################
class YoloV1Generator(keras.utils.Sequence):
    def __init__(self, data_dir, input_shape, batch_size, num_classes, num_boxes, transforms, grid=7, drop_remainder=False, shuffle=False):
        self.img_path_array = np.array(glob(data_dir + '/*.jpg'))
        self.input_shape = input_shape
        self.output_shape = (grid, grid, num_classes + (num_boxes*5))
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.grid = grid
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.transforms = transforms
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_path_array))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        if self.drop_remainder:
            return int(len(self.img_path_array) // self.batch_size)
        else:
            share = float(len(self.img_path_array) // self.batch_size)
            division_result = float(len(self.img_path_array) / self.batch_size)
            if division_result - share > 0.:
                return int(share + 1)
            else:
                return int(share)

    def __getitem__(self, index):
        if self.drop_remainder:
            indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        else:
            if (index + 1) * self.batch_size >= len(self.img_path_array):
                indexes = self.indexes[index * self.batch_size:]
            else:
                indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        img_path_array = self.img_path_array[indexes]
        x, y = self._get_data(img_path_array)
        return x, y

    def _get_data(self, img_path_array):
        cv2.setNumThreads(0)

        batch_images = np.zeros(
            shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            dtype=np.float32
        )

        batch_labels = np.zeros(
            shape=(self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]),
            dtype=np.float32
        )

        for i, img_path in enumerate(img_path_array):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label_path = img_path.replace('.jpg', '.txt')
            boxes = self._get_boxes(label_path)
            
            transformed = self.transforms(image=image, bboxes=boxes)

            batch_images[i] = transformed['image']
            batch_labels[i] = self._get_labels(transformed['bboxes'])

        return batch_images, batch_labels

    def _get_labels(self, boxes):
        # labels matrix = S*S*(B*5 + C)
        labels_matrix = np.zeros(self.output_shape)

        for box in boxes:
            # Get Class index, bbox info
            cls = int(box[-1])
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]

            # Start from grid position and calculate x, y
            loc = [self.grid * cy, self.grid * cx]
            loc_i = int(loc[0])
            loc_j = int(loc[1])
            y = loc[0] - loc_i
            x = loc[1] - loc_j

            if labels_matrix[loc_i, loc_j, self.num_classes] == 0: # confidence
                labels_matrix[loc_i, loc_j, cls] = 1 # class
                labels_matrix[loc_i, loc_j, self.num_classes+1:self.num_classes+5] = [x, y, w, h]
                labels_matrix[loc_i, loc_j, self.num_classes] = 1 # confidence

        return labels_matrix

    def _get_boxes(self, label_path):
        boxes = np.zeros((0, 5))
        with open(label_path, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                class_id, cx, cy, w, h = map(float, annot.split(' '))
                annotation = np.array([[cx, cy, w, h, class_id]])
                boxes = np.append(boxes, annotation, axis=0)

        return boxes


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
        self.dense_3 = keras.layers.Dense(units=7*7*(num_boxes*5 + num_classes), name='dense_3')
        self.yolo_v1_outputs = keras.layers.Reshape(target_shape=(7, 7, (num_boxes*5 + num_classes)), name='yolo_v1_outputs')

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
        x = self.dense_3(x) 
        return self.yolo_v1_outputs(x)
    
    def build_graph(self):
        x = self.backbone.input
        return keras.Model(inputs=x, outputs=self.call(x))

##################################
# YOLO v1 Loss Function
##################################
class YoloV1Loss(keras.losses.Loss):
    """YoloV1 Loss Function

    Attributes:
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
    ##################################
    # Setting up training parameters
    ##################################
    pwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    model_dir = os.path.join(pwd, "yolo_v1_models")
    data_dir = os.path.join(pwd, "data")
    names_path = os.path.join(pwd, "data/test.names")
    num_classes = 3
    num_boxes = 2
    batch_size = 1
    epochs = 200
    learning_rate = 0.001
    input_shape = (448, 448, 3)


    ##################################
    # Setting up datasets
    ##################################
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(),
        A.RandomResizedCrop(448, 448, (0.8, 1)),
        A.Normalize(0, 1)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
    
    test_transforms = A.Compose([
        A.Resize(448, 448),
        A.Normalize(0, 1)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))

    train_generator = YoloV1Generator(data_dir, input_shape, batch_size, num_classes, num_boxes, transforms=train_transforms)
    test_generator = YoloV1Generator(data_dir, input_shape, batch_size, num_classes, num_boxes, transforms=test_transforms)

    ##################################
    # Initializing and compiling model
    ##################################
    loss_fn = YoloV1Loss(num_classes, num_boxes)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    backbone = keras.applications.VGG16(include_top=False, input_shape=(None, None, 3))
    backbone.trainable = False
    model = YoloV1(num_classes, num_boxes, backbone)
    # model.build_graph().summary()

    # for layer in model.layers:
        # print(layer, layer.trainable)
    
    model.compile(loss=loss_fn, optimizer=optimizer)


    ##################################
    # Setting up callbacks
    ##################################
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "yolo_v1_best_model"),
            monitor="loss",
            save_best_only=True,
            # save_weights_only=True,
            verbose=1
        )
    ]


    ##################################
    # Training the model
    ##################################
    # model.fit(
    #     x=train_generator,
    #     epochs=epochs,
    #     verbose=1,
    #     callbacks=callbacks_list
    # )

    ##################################
    # Evaluating the model
    ##################################
    model_path = os.path.join(model_dir, "yolo_v1_best_model")
    evaluate_model = keras.models.load_model(model_path, compile=False)
    
    from utils import MeanAveragePrecision
    map_metric = MeanAveragePrecision(num_classes, num_boxes)
    map_metric_np = MeanAveragePrecisionNumpy(num_classes, num_boxes)
    
    for idx in range(test_generator.__len__()):
        batch_x, batch_y = test_generator.__getitem__(idx)
        
        predictions = evaluate_model(batch_x, training=False)
        
        map_metric.update_state(batch_y, predictions)
        map_metric_np.update_state(np.array(batch_y, dtype=np.float32), np.array(predictions, dtype=np.float32))

    map = map_metric.result()
    print(f'mAP: {map:.4f}')
    
    map = map_metric_np.result()
    print(f'mAP: {map:.4f}')
