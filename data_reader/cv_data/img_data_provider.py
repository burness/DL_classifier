#coding:utf-8
#-*-coding:utf-8
from __future__ import absolute_import
import sys
sys.path.append("..")
from data_reader.data_provider import DataReader
from data_reader.utils import *
import numpy as np

class ImgReader(DataReader):
    """
    图像数据读取类
    """
    def __init__(self, img_data_path, batch_size, **auxiliary_params):
        """
        ImgReader初始化函数
        :param img_data_path: 图像数据路径
        :param batch_size: 图像数据batch大小
        :param epoch: epoch数量
        :auxiliary_params: 辅助参数字典, type:dict
        """
        super(ImgReader, self).__init__(img_data_path, batch_size)
        self.data_type = "image_data"
        self.auxiliary_params = auxiliary_params
