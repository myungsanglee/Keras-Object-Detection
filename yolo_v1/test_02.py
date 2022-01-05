import os

import numpy as np
import cv2
import albumentations as A


pwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
label_path = os.path.join(pwd, "data/test.txt")
image_path = os.path.join(pwd, "data/test.jpg")

boxes = np.zeros((0, 5))
with open(label_path, 'r') as f:
    annotations = f.read().splitlines()
    for annot in annotations:
        class_id, cx, cy, w, h = map(float, annot.split(' '))
        print(class_id, cx, cy, w, h)
        annotation = np.array([[cx, cy, w, h, class_id]])
        boxes = np.append(boxes, annotation, axis=0)

print(boxes)

image = cv2.imread(image_path)

transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(),
    A.RandomResizedCrop(448, 448, (0.8, 1)),
    A.Normalize(0, 1)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))

transformed = transforms(image=image, bboxes=boxes)

image = transformed['image']
print(type(image), image.shape, image.dtype)
label = transformed['bboxes']

cv2.imshow("test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()