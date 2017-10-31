import os
import numpy as np
import tensorflow as tf

import reader
import model

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
RATIO = 0.2
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 0.0003

reader = Reader('./data/train/', ['cat', 'dog'])
r.split_image_list(RATIO)

train_img, train_label = r.get_train_batch(IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
val_img, val_label = r.get_val_batch(IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE])
