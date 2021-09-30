from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from glob import glob
from datetime import datetime
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from dataset import YoloV1Generator, YoloV1Generator2
from model import yolov1, mobilenet_v2_yolo_v1
from loss import YoloV1Loss
from metric import MeanAveragePrecision, MeanAveragePrecision2


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
# Variables
##################################
"""
S is split size of image (in paper 7),
B is number of boxes (in paper 2),
C is number of classes (in paper and VOC dataset is 20),
"""
S = 7
B = 2
C = 20

input_shape = (448, 448, 3)
output_shape = (S, S, C + (B * 5))

lr = 0.0001
batch_size = 64
training_epochs = 1000

model_name = "mobilenet_v2_yolo_v1"

# path variables
fmt = "%Y-%m-%d %H:%M:%S"
train_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/train"
val_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/val"
test_dir = "/home/fssv2/myungsang/datasets/voc_2007/yolo_format/test"
cur_time = datetime.now().strftime(fmt)
cur_dir = os.getcwd()

save_model_dir = os.path.join(cur_dir, "saved_models", cur_time)
os.makedirs(save_model_dir, exist_ok=True)

tensorboard_dir = os.path.join(cur_dir, "tensorboard_logs", cur_time)
os.makedirs(tensorboard_dir, exist_ok=True)


##################################
# Get Dataset Generator
##################################
train_generator = YoloV1Generator2(train_dir,
                                   input_shape=input_shape,
                                   batch_size=batch_size,
                                   drop_remainder=True,
                                   augment=True,
                                   shuffle=True)

val_generator = YoloV1Generator2(val_dir,
                                 input_shape=input_shape,
                                 batch_size=batch_size,
                                 drop_remainder=True,
                                 augment=False,
                                 shuffle=True)

test_generator = YoloV1Generator2(val_dir,
                                  input_shape=input_shape,
                                  batch_size=batch_size,
                                  drop_remainder=False,
                                  augment=False,
                                  shuffle=False)


##################################
# YOLO v1 Model
##################################
# model = yolov1(input_shape, output_shape)
model = mobilenet_v2_yolo_v1(input_shape, output_shape)
model.summary()


##################################
# Loss & optimizer
##################################
yolo_loss = YoloV1Loss()
optimizer = keras.optimizers.Adam(learning_rate=lr)

##################################
# Tensorboard Writer
##################################
valid_writer = tf.summary.create_file_writer(tensorboard_dir + '/validation')
test_writer = tf.summary.create_file_writer(tensorboard_dir + '/test')


##################################
# Callbacks
##################################
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.5,
                                              patience=50,
                                              verbose=1,
                                              mode='min',
                                              min_lr=0.00001)

save_model = keras.callbacks.ModelCheckpoint(filepath=save_model_dir + '/' + model_name + '_{epoch:05d}.h5',
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='min',
                                             save_freq='epoch')

tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, profile_batch=0)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=1000,
                                           verbose=1,
                                           mode='min')


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, test_generator, training=True, test_writer=None, writer_name='val_mAP', monitor='val_loss', mode='min'):
        super(CustomCallback, self).__init__()
        self.test_generator = test_generator
        self.monitor = monitor
        self.test_writer = test_writer
        self.writer_name = writer_name
        self.mode = mode
        self.training = training
        self.map_metric = MeanAveragePrecision2()

        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_test_end(self, logs=None):
        if not self.training:
            self._calculate_map(epoch=0)

    def on_epoch_end(self, epoch, logs=None):
        current = logs[self.monitor]

        if (epoch + 1) > 50:
            if self.monitor_op(current, self.best):
                print('\nEpoch {:05d}: {:s} improved from {:0.5f} to {:0.5f}, '
                      'calculate mean average precision'.format(epoch + 1, self.monitor, self.best, current))
                self._calculate_map(epoch=epoch)
                self.best = current
            elif (epoch + 1) % 10 == 0:
                print('\nEpoch {:05d}: calculate mean average precision'.format(epoch + 1))
                self._calculate_map(epoch=epoch)

    def _calculate_map(self, epoch=None):
        print('Calculating mean average precision. It takes sometime.')
        start_time = time.time()
        for idx in tqdm(range(self.test_generator.__len__()), desc='updating...'):
            batch_x, batch_y = self.test_generator.__getitem__(idx)
            predictions = self.model(batch_x, training=False)
            self.map_metric.update_state(batch_y, predictions)

        map = self.map_metric.result()
        print('mAP: {:.4f}, taken_time: {:.4f}'.format(map, time.time() - start_time))

        self.map_metric.reset_states()

        if self.test_writer is not None:
            with self.test_writer.as_default():
                tf.summary.scalar(name=self.writer_name, data=map, step=epoch)


custom_callback = CustomCallback(test_generator=val_generator,
                                 training=True,
                                 test_writer=valid_writer)

callbacks = [save_model, reduce_lr, tensorboard, custom_callback]
# callbacks = [save_model]

##################################
# Train Model
##################################
model.compile(optimizer=optimizer, loss=yolo_loss)

model.fit(x=train_generator,
          epochs=training_epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=val_generator)


##################################
# Evaluate Model
##################################
# Get Best model path
model_list = glob(save_model_dir + '/*')
model_list = sorted(model_list)
best_model_path = model_list[-1]
print('best_model_path: ', best_model_path)

# Get best model
# best_model = yolov1(input_shape, output_shape)
best_model = mobilenet_v2_yolo_v1(input_shape, output_shape)
best_model.load_weights(best_model_path)

best_model.compile(optimizer=optimizer, loss=yolo_loss)
loss = model.evaluate(x=test_generator,
                      callbacks=[CustomCallback(test_generator=test_generator, training=False, test_writer=test_writer, writer_name='test_mAP')])
if test_writer is not None:
    with test_writer.as_default():
        tf.summary.scalar(name='test_loss', data=loss, step=0)
