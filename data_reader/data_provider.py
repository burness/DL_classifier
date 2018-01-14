#coding: utf-8
class DataReader(object):
    """
    读取数据类
    """
    def __init__(self, data_path, batch_size):
        """
        DataReader初始化函数
        :param data_path: 数据路径
        :param batch_size: 数据batch大小
        """
        self.data_path = data_path
        self.batch_size = batch_size
    
    def next_batch(self):
        """
        获取下一个batch数据
        """
        print "implemented by subclass"


    def set_cache_size(self, cache_size):
        """
        设置缓冲池大小
        :param cache_size: 缓冲池大小
        """
        self.cache_size = cache_size



    
