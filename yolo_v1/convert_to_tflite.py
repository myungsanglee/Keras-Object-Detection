from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np


def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 448, 448, 3)
        yield [data.astype(np.float32)]


if __name__ == "__main__":
    # saved_model_dir = "/home/fssv2/myungsang/my_projects/keras-object-detection/yolo_v1/saved_models/yolo_v1_00001.h5"
    # model = tf.keras.models.load_model(saved_model_dir, compile=False)

    # IMG_SHAPE = (448, 448, 3)
    #
    # # Create the base model from the pre-trained MobileNet V2
    # base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
    #                                                include_top=False,
    #                                                weights='imagenet')
    # base_model.trainable = False
    #
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(base_model.output)
    # x = tf.keras.layers.LeakyReLU(0.1)(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(units=5, activation='softmax')(x)
    # # x = tf.keras.layers.Reshape(target_shape=(1, 5))(x)
    # model = tf.keras.Model(base_model.input, x)
    # model.summary()

    # input_shape = (448, 448, 3)
    # output_shape = (7, 7, 30)
    # model = yolov1(input_shape, output_shape)
    # model.summary()

    model = tf.keras.models.load_model("./test.h5")
    model.summary()

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)  # path to the SavedModel directory
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_dataset
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    # Save the model.
    with open('test.tflite', 'wb') as f:
        f.write(tflite_model)
