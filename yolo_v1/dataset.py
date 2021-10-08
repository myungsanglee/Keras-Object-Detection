from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow import keras
import cv2
from glob import glob
from imgaug import augmenters as iaa
import imgaug as ia
from utils import get_tagged_img

##################################
# Generator
# YOLO 포맷 형식의 데이터셋 제너레이터
##################################
class YoloV1Generator(keras.utils.Sequence):
    def __init__(self, data_dir, input_shape, batch_size, S=7, B=2, C=20, augment=False, shuffle=False):
        self.img_path_array = np.array(glob(data_dir + '/*.jpg'))
        self.input_shape = input_shape
        self.output_shape = (S, S, C + (B*5))
        self.batch_size = batch_size
        self.S = S
        self.B = B
        self.C = C
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_path_array))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.img_path_array)) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        img_path_array = self.img_path_array[indexes]
        x, y = self.__data_gen(img_path_array)
        return x, y

    def __data_gen(self, img_path_array):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.float32)

        batch_labels = np.zeros(shape=(self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]),
                                dtype=np.float32)

        for i, img_path in enumerate(img_path_array):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
            image = image / 255.

            label_path = img_path.replace('.jpg', '.txt')
            label = self.__get_label_matrix(label_path)

            batch_images[i] = image
            batch_labels[i] = label

        return batch_images, batch_labels

    def __get_label_matrix(self, label_path):
        # label matrix = S*S*(B*5 + C)
        label_matrix = np.zeros(self.output_shape)

        # Get label data
        with open(label_path, 'r') as f:
            label_data = f.readlines()
        label_data = [data.strip() for data in label_data]

        for data in label_data:
            # Get data list of label, bbox info
            data_list = data.split(' ')
            data_list = [float(data) for data in data_list]

            # Get Class index, bbox info
            cls = int(data_list[0])
            x = data_list[1]
            y = data_list[2]
            w = data_list[3]
            h = data_list[4]

            # Start from grid position and calculate x, y
            loc = [self.S * y, self.S * x]
            loc_i = int(loc[0])
            loc_j = int(loc[1])
            y = loc[0] - loc_i
            x = loc[1] - loc_j

            if label_matrix[loc_i, loc_j, self.C] == 0: # confidence
                label_matrix[loc_i, loc_j, cls] = 1 # class
                label_matrix[loc_i, loc_j, self.C+1:self.C+5] = [x, y, w, h]
                label_matrix[loc_i, loc_j, self.C] = 1 # confidence

        return label_matrix


class YoloV1Generator2(keras.utils.Sequence):
    def __init__(self, data_dir, input_shape, batch_size, drop_remainder=False, S=7, B=2, C=20, augment=False, shuffle=False):
        self.img_path_array = np.array(glob(data_dir + '/*.jpg'))
        self.input_shape = input_shape
        self.output_shape = (S, S, C + (B*5))
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.S = S
        self.B = B
        self.C = C
        self.augment = augment
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
        x, y = self.__data_gen(img_path_array)
        return x, y

    def __data_gen(self, img_path_array):
        cv2.setNumThreads(0)
        if not self.augment:
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
                image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                image = image / 255.

                label_path = img_path.replace('.jpg', '.txt')
                label = self.__get_label_matrix(label_path)

                batch_images[i] = image
                batch_labels[i] = label

        else:
            batch_images = np.zeros(
                shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                dtype=np.uint8
            )

            batch_labels = np.zeros(
                shape=(self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]),
                dtype=np.float32
            )

            bbs = []

            for i, img_path in enumerate(img_path_array):
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                batch_images[i] = image_resized

                label_path = img_path.replace('.jpg', '.txt')
                ia_bbox_list = self.__get_ia_bbox_list(label_path)
                bbs.append(ia_bbox_list)

            images_aug, bbs_aug = self.__augment_images_and_bboxes(batch_images, bbs)

            for i, ia_bboxes in enumerate(bbs_aug):
                batch_labels[i] = self.__get_label_matrix_from_ia_bbox(ia_bboxes)

            batch_images = images_aug.astype(np.float32) / 255.

        return batch_images, batch_labels

    def __get_label_matrix_from_ia_bbox(self, ia_bboxes):
        label_matrix = np.zeros(self.output_shape)

        for ia_bbox in ia_bboxes:
            # Get Class index, bbox info
            cls = ia_bbox.label
            xmin = ia_bbox.x1
            ymin = ia_bbox.y1
            xmax = ia_bbox.x2
            ymax = ia_bbox.y2

            x = ((xmin + xmax) / 2.) / self.input_shape[0]
            if x >= 1.: continue
            y = ((ymin + ymax) / 2.) / self.input_shape[1]
            if y >= 1.: continue
            w = (xmax - xmin) / self.input_shape[0]
            h = (ymax - ymin) / self.input_shape[1]

            # Start from grid position and calculate x, y
            loc = [self.S * y, self.S * x]
            loc_i = int(loc[0])
            loc_j = int(loc[1])
            y = loc[0] - loc_i
            x = loc[1] - loc_j

            if label_matrix[loc_i, loc_j, self.C] == 0: # confidence
                label_matrix[loc_i, loc_j, cls] = 1 # class
                label_matrix[loc_i, loc_j, self.C+1:self.C+5] = [x, y, w, h]
                label_matrix[loc_i, loc_j, self.C] = 1 # confidence

        return label_matrix

    def __get_ia_bbox_list(self, label_path):
        dst = []

        # Get label data
        with open(label_path, 'r') as label_file:
            label_data = label_file.readlines()
        label_data_list = [y.split(' ') for y in [x.strip() for x in label_data]]

        for data in label_data_list:
            class_idx = int(data[0])
            x = float(data[1]) * self.input_shape[0]
            y = float(data[2]) * self.input_shape[1]
            w = float(data[3]) * self.input_shape[0]
            h = float(data[4]) * self.input_shape[1]

            xmin = x - (w / 2.)
            ymin = y - (h / 2.)
            xmax = x + (w / 2.)
            ymax = y + (h / 2.)

            dst.append(ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=class_idx))

        return dst

    def __augment_images_and_bboxes(self, images, bboxes):
        seq = iaa.Sequential(
            [
                iaa.SomeOf((0, 7),  # Apply 1 to <max> given augmenters
                           # iaa.SomeOf(1,  # Apply 1 of given augmenters
                           [
                               iaa.Identity(),  # no change

                               # imgaug.augmenters.convolutional
                               iaa.Sharpen(),

                               # imgaug.augmenters.flip
                               iaa.Fliplr(), # Vertical
                               iaa.Flipud(0.5), # Horizontal

                               # imgaug.aumenters.color
                               iaa.MultiplyBrightness((0.5, 1.5)),
                               iaa.MultiplySaturation((0.5, 1.5)),
                               iaa.MultiplyHue((0.5, 1.5))

                               # imgaug.augmenters.geometric
                               # iaa.ScaleX((1.0, 1.2)),
                               # iaa.ScaleY((1.0, 1.2)),
                               # iaa.TranslateX(percent=(-0.2, 0.2)),
                               # iaa.TranslateY(percent=(-0.2, 0.2)),

                               # iaa.Solarize(threshold=0),  # inverts all pixel values above a threshold
                               # iaa.HistogramEqualization(),
                               # iaa.Posterize(nb_bits=(1, 8)),
                               # iaa.GammaContrast(gamma=(0.5, 2.0)),
                               # iaa.Rot90(k=(2, 3)),
                           ]
                           )
            ]
        )
        return seq(images=images, bounding_boxes=bboxes)

    def __get_label_matrix(self, label_path):
        # label matrix = S*S*(B*5 + C)
        label_matrix = np.zeros(self.output_shape)

        # Get label data
        with open(label_path, 'r') as f:
            label_data = f.readlines()
        label_data = [data.strip() for data in label_data]

        for data in label_data:
            # Get data list of label, bbox info
            data_list = data.split(' ')
            data_list = [float(data) for data in data_list]

            # Get Class index, bbox info
            cls = int(data_list[0])
            x = data_list[1]
            y = data_list[2]
            w = data_list[3]
            h = data_list[4]

            # Start from grid position and calculate x, y
            loc = [self.S * y, self.S * x]
            loc_i = int(loc[0])
            loc_j = int(loc[1])
            y = loc[0] - loc_i
            x = loc[1] - loc_j

            if label_matrix[loc_i, loc_j, self.C] == 0: # confidence
                label_matrix[loc_i, loc_j, cls] = 1 # class
                label_matrix[loc_i, loc_j, self.C+1:self.C+5] = [x, y, w, h]
                label_matrix[loc_i, loc_j, self.C] = 1 # confidence

        return label_matrix


def test(img, label_path):
    width = img.shape[1]
    height = img.shape[0]
    with open('/home/fssv2/myungsang/datasets/voc_2007/voc.names', 'r') as f:
        class_name_list = f.readlines()
    class_name_list = [x.strip() for x in class_name_list]
    with open(label_path, 'r') as label_file:
        label_data = label_file.readlines()
    label_data_list = [x.strip() for x in label_data]
    label_data_list = [x.split(' ') for x in label_data_list]
    for data in label_data_list:
        cls = int(data[0])
        x = float(data[1]) * width
        y = float(data[2]) * height
        w = float(data[3]) * width
        h = float(data[4]) * height

        xmin = int(x - (w / 2.))
        ymin = int(y - (h / 2.))
        xmax = int(x + (w / 2.))
        ymax = int(y + (h / 2.))

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255))
        img = cv2.circle(img, (int(x), int(y)), radius=2, color=(0, 0, 255))
        img = cv2.putText(img, "{:s}".format(class_name_list[cls]), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=(0, 0, 255))
    for idx in range(6):
        a = int(448 * ((idx + 1) / 7.))
        img = cv2.line(img, (a, 0), (a, height), color=(255, 0, 255))
        img = cv2.line(img, (0, a), (width, a), color=(255, 0, 255))

    return img


if __name__ == "__main__":
    train_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/train"
    val_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/val"
    test_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/test"
    input_shape = (448, 448, 3)
    batch_size = 1

    # tmp_generator = YoloV1Generator(test_dir, input_shape, batch_size)
    generator_1 = YoloV1Generator2(test_dir, input_shape, batch_size, augment=True, shuffle=False)
    generator_2 = YoloV1Generator2(test_dir, input_shape, batch_size, augment=False, shuffle=False)
    img_path_array = generator_2.img_path_array

    cv2.namedWindow('Aug')
    cv2.resizeWindow('Aug', 448, 448)
    cv2.namedWindow('Origin')
    cv2.resizeWindow('Origin', 448, 448)
    cv2.namedWindow('Origin_2')
    cv2.resizeWindow('Origin_2', 448, 448)

    for idx in range(generator_1.__len__()):
        x_aug, y_aug = generator_1.__getitem__(idx)
        x_origin, y_origin = generator_2.__getitem__(idx)
        img_path = img_path_array[idx]
        label_path = img_path.replace('.jpg', '.txt')

        x_aug = cv2.cvtColor(x_aug[0], cv2.COLOR_RGB2BGR)
        x_origin = cv2.cvtColor(x_origin[0], cv2.COLOR_RGB2BGR)
        x_origin_2 = x_origin.copy()

        x_aug = get_tagged_img(x_aug, y_aug, prediction=False)
        x_origin = get_tagged_img(x_origin, y_origin, prediction=False)
        x_origin_2 = test(x_origin_2, label_path)

        cv2.imshow('Aug', x_aug)
        cv2.imshow('Origin', x_origin)
        cv2.imshow('Origin_2', x_origin_2)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
