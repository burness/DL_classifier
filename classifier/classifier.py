#coding:utf-8

class Classifier(object):
    """
    分类器实现父类
    """
    def __init__(self, classifier_name, data_reader, type, **classifier_params):
        """
        分类器初始化函数
        :param classifier_name: 分类器名
        :param data_reader: 数据读取生成器
        :param type: 分类器类别
        :param classifier_params: 分类器参数
        """
        self.classifier_name = classifier_name
        self.data_reader = data_reader
        self.type = type
        self.classifier_params = classifier_params
    
    def define_model(self, network_arc):
        """
        定义网络结构
        :param network_arc: 神经网络结构
        """
        print "Implemented by subclass"

    def fit(self):
        """
        模型训练函数
        """
        print "Implemented by subclass"

    def evaluate(self):
        """
        模型评估函数
        """
        print "Implemented by subclass"

    def export_model(self):
        """
        模型导出函数：导出模型直接可导入TensorFlow Serving
        """
        print "Implemented by subclass"


