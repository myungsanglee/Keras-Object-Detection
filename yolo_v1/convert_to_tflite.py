import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

from yolo_v1 import decode_predictions

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 448, 448, 3)
        yield [data.astype(np.float32)]


if __name__ == "__main__":
    # # Convert the model #1
    # # converter = tf.lite.TFLiteConverter.from_keras_model(model)  # path to the SavedModel directory
    # converter = tf.lite.TFLiteConverter.from_saved_model("test")  # path to the SavedModel directory
    # # This enables quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # This sets the representative dataset for quantization
    # converter.representative_dataset = representative_dataset
    # # This ensures that if any ops can't be quantized, the converter throws an error
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    # converter.target_spec.supported_types = [tf.int8]
    # # These set the input and output tensors to uint8 (added in r2.3)
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # tflite_model = converter.convert()
    
    # Convert the model #2
    pwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    model_dir = os.path.join(pwd, "yolo_v1_models", "yolo_v1_best_model")
    model = keras.models.load_model(model_dir, compile=False)
    model.summary()
    # tmp_inputs = keras.Input(shape=(448, 448, 3))
    # model(tmp_inputs, training=False)
    
    # new_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    # new_model.summary()
    # converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    # with open('test.tflite', 'wb') as f:
    #     f.write(tflite_model)
