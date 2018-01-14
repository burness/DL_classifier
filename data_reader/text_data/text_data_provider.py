#-*-coding:utf-8
from __future__ import absolute_import
import sys
sys.path.append("..")
from data_reader.data_provider import DataReader
from data_reader.utils import *
import numpy as np

class TextReader(DataReader):
    """
    文本数据读取类
    """
    def __init__(self, text_data_path, batch_size, **auxiliary_params):
        """
        TextReader初始化函数
        :param text_data_path: 文本数据路径
        :param batch_size: 数据batch大小
        :param epoch: epoch数量
        :auxiliary_params: 辅助参数字典，type: dict
        """
        super(TextReader, self).__init__(text_data_path, batch_size)
        self.data_type = "text_data"
        self.auxiliary_params = auxiliary_params


    def __read(self):
        """
        获取文本数据
        """
        fread = open_read(self.data_path)
        line_list = fread.readlines()
        fread.close()
        return line_list
    
    def batch_iter(self, shuffle=True):
        """
        获取下一个batch文本数据
        TODO: 理解keras的fit_generator的generator需求，注意epoch逻辑怎么设置，应该不需要再此处
        :param shuffle: 是否打乱数据顺序
        :return num_batches_per_epoch: 每个epoch batch数目
        :return data_generator(): 数据生成器
        """
        line_list = self.__read()
        label_index = self.auxiliary_params["label_index"] if self.auxiliary_params.has_key("label_index") else 0
        data_list = [i.strip("\n").split("\t") for i in line_list]
        data_size = len(data_list)
        label = np.array([i[label_index] for i in data_list])
        feature = np.array([i[:label_index]+i[label_index+1:] for i in data_list])
        num_batches_per_epoch = int((data_size- 1) / self.batch_size) + 1
        def data_generator():
            while True:
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffle_feature = feature[shuffle_indices]
                    shuflle_label = label[shuffle_indices]
                else:
                    shuffle_feature = feature
                    shuflle_label = label
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * self.batch_size
                    end_index = min((batch_num + 1) * self.batch_size, data_size)
                    X, y = shuffle_feature[start_index: end_index], shuflle_label[start_index: end_index]
                    yield X, y
        return num_batches_per_epoch, data_generator()


    @property
    def text_data_path(self):
        """
        获取文本数据路径
        """
        return self.data_path

    @text_data_path.setter
    def text_data_path(self, text_data_path):
        """
        设置文本数据路径
        """
        self.data_path = text_data_path


    


    


    
