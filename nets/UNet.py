# -*- coding: utf-8 -*-
# @File : U-Net.py
# @Author: Runist
# @Time : 2021/6/25 16:09
# @Software: PyCharm
# @Brief: unet网络结构实现

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def UNet(input_shape, num_classes):
    """
    U-Net网络结构
    :param input_shape: 输入shape
    :param num_classes: 分类数量
    :return:
    """
    inputs = layers.Input(input_shape)

    # downsample
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # upsample
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(
        layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.Concatenate()([drop4, up6])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.Concatenate()([conv3, up7])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.Concatenate()([conv2, up8])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.Concatenate()([conv1, up9])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv9)
    outputs = layers.Conv2D(num_classes, 1, padding='same', kernel_regularizer=regularizers.l2(5e-4))(conv9)

    model = models.Model(inputs=inputs, outputs=outputs, name='U-Net')

    return model
