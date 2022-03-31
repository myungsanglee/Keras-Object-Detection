import os
import time

from tensorflow import keras
import cv2
import albumentations as A

from utils import get_tagged_img, decode_predictions, non_max_suppression
from dataset import YoloV1Generator

pwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
model_dir = os.path.join(pwd, "yolo_v1_models")
data_dir = os.path.join(pwd, "data")
names_path = os.path.join(pwd, "data/test.names")
num_classes = 3
num_boxes = 2
batch_size = 1
input_shape = (448, 448, 3)

model_path = os.path.join(model_dir, "yolo_v1_best_model")
model = keras.models.load_model(model_path, compile=False)
model.summary()

tmp_inputs = keras.Input(shape=(448, 448, 3), batch_size=1)
predictions = model(tmp_inputs, training=False)
boxes = keras.layers.Lambda(decode_predictions, arguments={"num_classes":num_classes, "num_boxes":num_boxes}, name="decode")(predictions)
inference_model = keras.Model(inputs=tmp_inputs, outputs=boxes)
inference_model.summary()

test_transforms = A.Compose([
        A.Resize(448, 448),
        A.Normalize(0, 1)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
test_generator = YoloV1Generator(data_dir, input_shape, batch_size, num_classes, num_boxes, transforms=test_transforms)
x_batch, y_batch = test_generator.__getitem__(0)

for i in range(5):
    start_1 = time.time()
    predictions = model(x_batch, training=False)
    nms_boxes_1 = non_max_suppression(decode_predictions(predictions, num_classes, num_boxes)[0])
    print(f"model time: {(time.time()-start_1)*1000:.2f} ms")
    
    start_2 = time.time()
    boxes = inference_model(x_batch, training=False)
    nms_boxes_2 = non_max_suppression(boxes[0])
    print(f"inference_model time: {(time.time()-start_2)*1000:.2f} ms")
    print(nms_boxes_2)

cv_img = cv2.cvtColor(x_batch[0], cv2.COLOR_RGB2BGR)
img_1 = get_tagged_img(cv_img, nms_boxes_1, names_path)
img_2 = get_tagged_img(cv_img, nms_boxes_2, names_path)

cv2.imshow("model", img_1)
cv2.imshow("inference model", img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
