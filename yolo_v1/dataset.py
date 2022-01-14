import os

import numpy as np
from tensorflow import keras
import cv2
from glob import glob
from imgaug import augmenters as iaa
import imgaug as ia
import albumentations as A

from utils import get_tagged_img, get_grid_tagged_img, decode_predictions, non_max_suppression


##################################
# Generator
# YOLO 포맷 형식의 데이터셋 제너레이터
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
        

if __name__ == "__main__":    
    pwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    data_dir = os.path.join(pwd, "data")
    names_path = os.path.join(pwd, "data/test.names")
    
    input_shape = (448, 448, 3)
    batch_size = 1
    num_classes = 3
    num_boxes = 2
    
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(),
        A.RandomResizedCrop(448, 448, (0.8, 1)),
        A.Normalize(0, 1)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
    
    valid_transforms = A.Compose([
        A.Resize(448, 448),
        A.Normalize(0, 1)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))

    dataset_generator = YoloV1Generator(data_dir, input_shape, batch_size, num_classes, num_boxes, transforms=valid_transforms)

    for idx in range(dataset_generator.__len__()):
        batch_x, batch_y = dataset_generator.__getitem__(idx)
        
        decode_result = non_max_suppression(decode_predictions(batch_y, num_classes, num_boxes)[0])
        
        batch_x = cv2.cvtColor(batch_x[0], cv2.COLOR_RGB2BGR)
        batch_x_copy = batch_x.copy()
        
        tagged_img = get_tagged_img(batch_x, decode_result, names_path)
        grid_tagged_img = get_grid_tagged_img(batch_x_copy, decode_result, names_path)

        cv2.imshow('img', tagged_img)
        cv2.imshow('grid img', grid_tagged_img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
