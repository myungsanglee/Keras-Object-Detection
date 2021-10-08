from __future__ import absolute_import

import os
import tensorflow as tf
from tensorflow import keras

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
# YOLO v1 Model
##################################
"""
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding)
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


def cnn_block(input_tensor, kernel_size, filters, strides, padding):
    x = keras.layers.ZeroPadding2D(padding=padding)(input_tensor)
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.ReLU()(x)
    return x


def yolov1_backbone(input_tensor, architecture):
    x = input_tensor
    for data in architecture:
        if type(data) == tuple:
            x = cnn_block(x, data[0], data[1], data[2], data[3])

        elif type(data) == str:
            x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        elif type(data) == list:
            conv1 = data[0]
            conv2 = data[1]
            num_repeats = data[2]

            for _ in range(num_repeats):
                x = cnn_block(x, conv1[0], conv1[1], conv1[2], conv1[3])
                x = cnn_block(x, conv2[0], conv2[1], conv2[2], conv2[3])

    return x


def yolov1(input_shape, output_shape, architecture=architecture_config):
    # Input tensor
    input_tensor = keras.layers.Input(input_shape)

    # backbone
    x = yolov1_backbone(input_tensor, architecture)

    # neck
    x = keras.layers.Flatten()(x)

    # Fully Connected Layer(head)
    x = keras.layers.Dense(units=4096)(x)
    # x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.ReLU()(x)
    # output_tensor = keras.layers.Dense(units=(output_shape[0]*output_shape[1]*output_shape[2]))(x)
    x = keras.layers.Dense(units=(output_shape[0]*output_shape[1]*output_shape[2]))(x)
    output_tensor = keras.layers.Reshape(target_shape=output_shape)(x)

    return keras.Model(input_tensor, output_tensor)


def mobilenet_v2_yolo_v1(input_shape, output_shape):
    # Input tensor
    input_tensor = keras.layers.Input(input_shape)

    # backbone
    backbone = keras.applications.MobileNetV2(include_top=False,
                                              weights=None,
                                              input_tensor=input_tensor)

    # YOLO V1 Head
    # Conv Layers
    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(backbone.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # Fully Conn Layers
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(units=4096)(x)

    x = keras.layers.Dropout(0.5)(x)

    output_tensor = keras.layers.Dense(units=(output_shape[0]*output_shape[1]*output_shape[2]))(x)

    return keras.Model(input_tensor, output_tensor)


def test_model(input_shape, output_shape):
    # Input tensor
    input_tensor = keras.layers.Input(input_shape)

    # backbone
    backbone = keras.applications.MobileNetV2(include_top=False,
                                              weights='imagenet',
                                              input_tensor=input_tensor,
                                              pooling='avg')

    # Fully Connected Layer(head)
    x = keras.layers.Dense(units=4096)(backbone.output)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(units=(output_shape[0]*output_shape[1]*output_shape[2]))(x)
    output_tensor = keras.layers.Reshape(target_shape=output_shape)(x)

    return keras.Model(input_tensor, output_tensor)


def vgg16_yolo_v1(input_shape, output_shape):
    # Input tensor
    input_tensor = keras.layers.Input(input_shape)

    # backbone
    backbone = keras.applications.VGG16(include_top=False,
                                        weights='imagenet',
                                        input_tensor=input_tensor)

    # YOLO V1 Head
    # Conv Layers
    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(backbone.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # x = keras.layers.ZeroPadding2D()(x)
    # x = keras.layers.LocallyConnected2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    # x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.ReLU()(x)

    # Fully Conn Layers
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(units=512)(x)
    x = keras.layers.Dense(units=1024)(x)

    x = keras.layers.Dropout(0.5)(x)

    output_tensor = keras.layers.Dense(units=(output_shape[0]*output_shape[1]*output_shape[2]))(x)

    return keras.Model(input_tensor, output_tensor)

if __name__ == "__main__":
    input_shape = (448, 448, 3)
    output_shape = (7, 7, 30)
    # model = yolov1(input_shape, output_shape)
    model = mobilenet_v2_yolo_v1(input_shape, output_shape)
    # model = vgg16_yolo_v1(input_shape, output_shape)
    model.summary()
    # model.save("test.h5")

    for layer in model.layers:
        print(layer.trainable)

    # new_model = keras.Model(model.input, model.layers[-2].output)
    # new_model.summary()

    # input_tensor = keras.layers.Input(input_shape)
    # backbone = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # backbone.summary()