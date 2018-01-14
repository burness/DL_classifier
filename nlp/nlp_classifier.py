#-*-coding:utf-8
import sys
sys.path.append("..")

from classifier.classifier import Classifier

class NLPClassifier(Classifier):
    """
    文本分类器函数
    """
    def __init__(self, nlp_classifier_name, text_data_reader, type="nlp", **classifier_params):
        """
        文本分类器初始化函数
        :param nlp_classifier_name: 文本分类器名
        :param data_reader: 数据读取生成器
        :param type: 分类器类别，nlp:文本分类器
        :param classifier_params: 分类器参数
        """
        super(NLPClassifier, self).__init__(nlp_classifier_name, text_data_reader, type, classifier_params)

    def define_model(self, network_arc):
        """
        定义网络结构
        :param network_arc: 神经网络结构
        """
        if isinstance(network_arc, str):
            # 通过网络结构名定义网络
            from importlib import import_module
            net = import_module('network.'+network_arc)
            self.network_arc = net.get_symbol(**vars(self.classifier_params))
        else:
            print "unsupported type of network_arc"
            return

    def fit(self):
        """
        模型训练函数
        """
        pass

    def evaluate(self):
        """
        模型评估函数
        """
        pass

    def export_model(self):
        """
        模型导出函数：导出模型直接导入TensorFlow Serving
        """
        pass
        
        