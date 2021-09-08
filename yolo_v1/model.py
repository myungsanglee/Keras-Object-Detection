from __future__ import absolute_import

import tensorflow as tf
from tensorflow import keras

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
    x = keras.layers.LeakyReLU(0.1)(x)
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
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dense(units=(output_shape[0]*output_shape[1]*output_shape[2]))(x)
    output_tensor = keras.layers.Reshape(target_shape=output_shape)(x)

    return keras.Model(input_tensor, output_tensor)
    # return (input_tensor, output_tensor)


if __name__ == "__main__":
    input_shape = (224, 224, 3)
    output_shape = (7, 7, 30)
    model = yolov1(input_shape, output_shape)
    model.summary()
