import os
import numpy as np
import csv
import random
import tensorflow as tf
import pandas as pd
import ImageOp


def processlabel(label, cato=2, delta1=0, delta2=0):
    softmaxlabel = np.zeros(len(label) * cato, dtype=np.float32).reshape(len(label), cato)
    for i in range(0, len(label)):
        if int(label[i]) == 0:
            softmaxlabel[i, 0] = 1 - delta1
            softmaxlabel[i, 1] = delta1
        if int(label[i]) == 1:
            softmaxlabel[i, 0] = delta2
            softmaxlabel[i, 1] = 1 - delta2
    return softmaxlabel


def feature(img, block_size, block_dim, fealen):
    img = ImageOp.rescale(img)
    feaarray = np.empty(fealen * block_dim * block_dim).reshape(fealen, block_dim, block_dim)
    blocked = ImageOp.cutblock(img, block_size, block_dim)
    for i in range(0, block_dim):
        for j in range(0, block_dim):
            featemp = ImageOp.subfeature(blocked[i, j], fealen)
            feaarray[:, i, j] = featemp
    return feaarray


def forward(inputs, training=True, reuse=False, scope='model', flip=False):
    if flip:
        inputs = tf.image.random_flip_left_right(inputs)
        inputs = tf.image.random_flip_up_down(inputs)

    with tf.variable_scope(scope, reuse=reuse):
        # 使用tf.keras.layers代替slim模块
        net = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform())(inputs)
        net = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)
        net = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(net)
        net = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(250, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(net)
        if training:
            net = tf.keras.layers.Dropout(0.5)(net)
        net = tf.keras.layers.Dense(2, activation=None)(net)  # 输出层不使用激活函数

    return net

# # tensorflow1.9
# def forward(input, is_training=True, reuse=False, scope='model', flip=False):
#     if flip == True:
#         input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
#         input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)
#
#     with tf.variable_scope(scope, reuse=reuse):
#         with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
#                             weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                             biases_initializer=tf.constant_initializer(0.0)):
#             net = slim.conv2d(input, 16, [3, 3], scope='conv1_1')
#             net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
#             net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
#             net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
#             net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
#             net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
#             net = slim.flatten(net)
#             w_init = tf.contrib.layers.xavier_initializer(uniform=False)
#             net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
#             net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
#             predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
#     return predict
