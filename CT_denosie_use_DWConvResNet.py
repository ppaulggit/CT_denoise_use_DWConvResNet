# -*- coding: utf-8 -*-

import tflearn
from tflearn.layers.conv import conv_2d, grouped_conv_2d
from tflearn.layers.core import input_data
from tflearn.layers.estimator import regression
from tflearn import batch_normalization, relu
from tflearn.objectives import mean_square
import numpy as np
import os
import cv2
import random
import time
import math
import tensorflow as tf


def Channel_Shuffle(input, num_groups, reuse=False, scope=None, name='ChannelShuffle'):
    with tf.variable_scope(scope, default_name=name, values=[input],
                           reuse=reuse) as scope:
        n, h, w, c = input.get_shape().as_list()
        input_reshaped = tf.reshape(input, [-1, h, w, num_groups, c//num_groups])
        input_transposed = tf.transpose(input_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(input_transposed, [-1, h, w, c])
        return output


def resnet_dwconv_block(input, channel_multiplier, outChannels,
                        nb, filter_size=3, regularizer='L2',
                        weights_init='variance_scaling',
                        weight_decay=0.0001, scope=None,
                        reuse=False, name='resnet_dwconv_block'):
    res_unit = input
    with tf.variable_scope(scope, default_name=name, values=[input],
                           reuse=reuse) as scope:
        name = scope.name
        for i in range(nb):
            identity = res_unit
            res_unit = batch_normalization(res_unit)
            res_unit = relu(res_unit)
            res_unit = grouped_conv_2d(res_unit, channel_multiplier,
                                       filter_size, weights_init=weights_init,
                                       regularizer=regularizer,
                                       weight_decay=weight_decay)
            res_unit = batch_normalization(res_unit)
            res_unit = relu(res_unit)
            res_unit = conv_2d(res_unit, outChannels, 1,
                               weights_init=weights_init,
                               regularizer=regularizer,
                               weight_decay=weight_decay)
            res_unit = Channel_Shuffle(res_unit, 8)
            res_unit = batch_normalization(res_unit)
            res_unit = relu(res_unit)
            res_unit = grouped_conv_2d(res_unit, channel_multiplier,
                                       filter_size, weights_init=weights_init,
                                       regularizer=regularizer,
                                       weight_decay=weight_decay)
            res_unit = batch_normalization(res_unit)
            res_unit = relu(res_unit)
            res_unit = conv_2d(res_unit, outChannels, 1,
                               weights_init=weights_init,
                               regularizer=regularizer,
                               weight_decay=weight_decay)
            if identity.get_shape().as_list()[-1] != res_unit.get_shape().as_list()[-1]:
                num_c = int(res_unit.get_shape().as_list()[-1])
                identity = conv_2d(identity, num_c, 1,
                                   weights_init=weights_init,
                                   regularizer=regularizer,
                                   weight_decay=weight_decay)
            res_unit = res_unit + identity

    return res_unit

img_size_w = 256
img_size_h = 256
num_channel = 1
LR = 0.0001
netname = 'DWConvResNet'
idx = '2'

MODEL_NAME = 'CT_denoise-{}-{}.model'.format(netname, idx)

tf.reset_default_graph()

input = input_data(shape=[None, img_size_w, img_size_h, num_channel], name='input')
net = conv_2d(input, 64, 3, weights_init='variance_scaling',
              regularizer='L2', weight_decay=0.0001)

net = resnet_dwconv_block(net, 1, 64, 2)
net = resnet_dwconv_block(net, 1, 128, 2)
net = resnet_dwconv_block(net, 1, 256, 4)
net = resnet_dwconv_block(net, 1, 128, 2)
net = resnet_dwconv_block(net, 1, 64, 2)

net = batch_normalization(net)

net = conv_2d(net, 1, 3, weights_init='variance_scaling',
              regularizer='L2', weight_decay=0.0001)
net = regression(net, optimizer='adam', learning_rate=LR,
                 loss=mean_square, name='target')
model = tflearn.DNN(net, clip_gradients=0., tensorboard_dir='log')

rec_file_name = 'image_data_R_4'
I_file_name = 'image_data_I_4'

train_noise_data_file = '{}_train_{}_{}.npy'.format(rec_file_name, img_size_h, img_size_w)
val_noise_data_file = '{}_val_{}_[].npy'.format(rec_file_name, img_size_h, img_size_w)
target_train_data_file = '{}_train_{}_{}.npy'.format(I_file_name, img_size_h, img_size_w)
target_val_data_file = '{}_val_{}_{}.npy'.format(I_file_name, img_size_h, img_size_w)

target_train_data = np.load(target_train_data_file)
target_train = np.array([i[0] for i in target_train_data]).reshape(-1, img_size_w, img_size_h, num_channel)
target_train = target_train

train_noise_data = np.load(train_noise_data_file)
train_noise = np.array([i[0] for i in train_noise_data]).reshape(-1, img_size_w, img_size_h, num_channel)
train_noise = train_noise

target_val_data = np.load(target_val_data_file)
target_val = np.array([i[0] for i in target_val_data]).reshape(-1, img_size_w, img_size_h, num_channel)
target_val = target_val

val_noise_data = np.load(val_noise_data_file)
val_noise = np.array([i[0] for i in val_noise_data]).reshape(-1, img_size_w, img_size_h, num_channel)
val_noise = val_noise

start_time = time.time()
model.fit({'input': train_noise}, {'target': target_train},
          validation_set=({'input': val_noise}, {'target': target_val}),
          n_epoch=10, batch_size=4, show_metric=True, run_id=MODEL_NAME)
duration = time.time() - start_time
print(duration/3600)

model.save(MODEL_NAME)
