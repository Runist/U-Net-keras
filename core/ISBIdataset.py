# -*- coding: utf-8 -*-
# @File : ISBIdataset.py
# @Author: Runist
# @Time : 2021/5/10 12:17
# @Software: PyCharm
# @Brief: ISBI数据集读取脚本
from core.dataset import Dataset
import numpy as np
import random
import tensorflow as tf
from PIL import Image
import os
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class ISBIdataset(Dataset):
    def __init__(self, annotation_path, batch_size=4, target_size=(512, 512), num_classes=1, aug=False):
        super().__init__(target_size, num_classes)
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.annotation_path = annotation_path
        self.aug = aug
        self.set_image_info()

    def __len__(self):
        return len(self.image_info)

    def set_image_info(self):
        """
        继承自Dataset类，需要实现对输入图像路径的读取和mask路径的读取，且存储到self.image_info中
        :return:
        """
        self.image_info = []
        image_dir = "{}/image".format(self.annotation_path)
        mask_dir = "{}/label".format(self.annotation_path)
        for img in os.listdir(image_dir):
            image_path = os.path.join(image_dir, img)
            mask_path = os.path.join(mask_dir, img)
            self.image_info.append({"image_path": image_path, "mask_path": mask_path})

    def read_mask(self, image_id, one_hot=False):
        """
        读取mask，并转换成语义分割需要的结构。
        :param image_id: 图像的id号
        :param one_hot: 是否转成one hot的形式
        :return: image
        """
        mask_path = self.image_info[image_id]["mask_path"]
        image = Image.open(mask_path)
        image = np.array(image)
        # 镜像映射
        # image = cv.copyMakeBorder(image, 30, 30, 30, 30, cv.BORDER_REFLECT)

        image = image / 255
        image[image > 0.5] = 1
        image[image <= 0.5] = 0

        if one_hot:
            # 转为one hot形式的标签
            h, w = image.shape[:2]
            mask = np.zeros((h, w, self.num_classes), np.uint8)

            for c in range(1, self.num_classes):
                m = np.argwhere(image == c)

                for row, col in m:
                    mask[row, col, c] = 1

            return mask

        return image

    def read_image(self, image_id):
        """
        读取图像
        :return: image
        """
        image_path = self.image_info[image_id]["image_path"]
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        # 边缘镜像映射
        # image = cv.copyMakeBorder(image, 30, 30, 30, 30, cv.BORDER_REFLECT)

        return image

    def resize_image_with_pad(self, image, target_size, pad_value=128.0):
        """
        resize图像，多余的地方用其他颜色填充
        :param image: 输入图像
        :param target_size: resize后图像的大小
        :param pad_value: 填充区域像素值
        :return: image_padded
        """
        image_h, image_w = image.shape[:2]
        input_h, input_w = target_size

        scale = min(input_h / image_h, input_w / image_w)

        image_h = int(image_h * scale)
        image_w = int(image_w * scale)

        dw, dh = (input_w - image_w) // 2, (input_h - image_h) // 2

        if pad_value == 0:
            # mask 用最近领域插值
            image_resize = cv.resize(image, (image_w, image_h), interpolation=cv.INTER_NEAREST)
        else:
            # image 用双线性插值
            image_resize = cv.resize(image, (image_w, image_h), interpolation=cv.INTER_LINEAR)

        image_padded = np.full(shape=[input_h, input_w], fill_value=pad_value)
        image_padded[dh: image_h+dh, dw: image_w+dw] = image_resize

        return image_padded

    def random_horizontal_flip(self, image, mask):
        """
        左右翻转图像
        :param image: 输入图像
        :param mask: 对应的mask图
        :return:
        """
        _, w = image.shape[:2]
        image = cv.flip(image, 1)
        mask = cv.flip(mask, 1)

        return image, mask

    def random_crop(self, image, mask):
        """
        随机裁剪
        :param image: 输入图像
        :param mask: 对应的mask图
        :return:
        """
        h, w = image.shape[:2]

        max_l_trans = 32
        max_u_trans = 32
        max_r_trans = w - 32
        max_d_trans = h - 32

        crop_xmin = int(random.uniform(0, max_l_trans))
        crop_ymin = int(random.uniform(0, max_u_trans))
        crop_xmax = int(random.uniform(max_r_trans, w))
        crop_ymax = int(random.uniform(max_d_trans, h))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
        mask = mask[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        return image, mask

    def random_translate(self, image, mask):
        """
        整图随机位移
        :param image: 输入图像
        :param mask: 对应的mask图
        :return:
        """

        h, w = image.shape[:2]

        max_l_trans = 32
        max_u_trans = 32
        max_r_trans = 32
        max_d_trans = 32

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])

        image = cv.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
        mask = cv.warpAffine(mask, M, (w, h), borderValue=(0, 0, 0))

        return image, mask

    def elastic_transform(self, image, mask, alpha, sigma, alpha_affine, random_state=None):
        """
        图片弹性形变
        :param image: 输入图片
        :param alpha: 控制变形强度的变形因子
        :param sigma: 类似bias的参数
        :param alpha_affine: 坐标位移的范围
        :param random_state:
        :return: 弹性形变后的图像
        """

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3

        # pts1: 仿射变换前的点(3个点)
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        # pts2: 仿射变换后的点
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                           size=pts1.shape).astype(np.float32)
        # 仿射变换矩阵
        M = cv.getAffineTransform(pts1, pts2)
        # 对image进行仿射变换.
        image = cv.warpAffine(image, M, shape_size[::-1], borderMode=cv.BORDER_REFLECT_101)
        mask = cv.warpAffine(mask, M, shape_size[::-1], borderMode=cv.BORDER_REFLECT_101)

        # 生成随机位移场
        # random_state.rand(*shape)会产生一个和shape一样的服从[0,1]均匀分布的矩阵
        # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        # 生成网格坐标
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # x+dx, y+dy
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        # 双线性插值
        image = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
        mask = map_coordinates(mask, indices, order=1, mode='constant').reshape(shape)

        return image, mask

    def parse(self, index):
        """
        tf.data的解析器
        :param index: 字典索引
        :return:
        """

        def get_data(i):
            image = self.read_image(i)
            mask = self.read_mask(i, one_hot=False)

            if random.random() < 0.5 and self.aug:
                image, mask = self.random_horizontal_flip(image, mask)
            if random.random() < 0.5 and self.aug:
                image, mask = self.random_crop(image, mask)
            if random.random() < 0.5 and self.aug:
                image, mask = self.random_translate(image, mask)
            if random.random() < 0.5 and self.aug:
                image, mask = self.elastic_transform(image, mask, image.shape[1] * 2,
                                                     image.shape[1] * 0.08,
                                                     image.shape[1] * 0.08)

            image = self.resize_image_with_pad(image, self.target_size, pad_value=0.)
            mask = self.resize_image_with_pad(mask, self.target_size, pad_value=0.)

            return image, mask

        image, mask = tf.py_function(get_data, [index], [tf.float32, tf.float32])
        h, w = self.target_size
        image.set_shape([h, w])
        mask.set_shape([h, w])

        return image, mask

    def tf_dataset(self):
        """
        用tf.data的方式读取数据，以提高gpu使用率
        :return: 数据集
        """
        index = [i for i in range(len(self))]
        # 这是GPU读取方式
        dataset = tf.data.Dataset.from_tensor_slices(index)

        dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat().batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if self.aug:
            dataset = dataset.shuffle(buffer_size=1)

        return dataset

