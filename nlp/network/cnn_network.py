#coding:utf-8
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.layers import Activation

def get_symbol(sequence_input, corpus_size, embedding_dim, embedding_matrix, max_sequence_length, class_num):
    """
    :param sequence_input: 序列输入
    :param corpus_size: 语料中所有词的数量
    :param embedding_dim: 语料中embedding维度
    :param embedding_matrix: 语料中词对应的向量表示矩阵
    :param max_sequence_length: 语料中最大序列长度
    :param class_num: 分类器类目长度
    """
    embedding_layer = Embedding(corpus_size,
                embedding_dim,
                weights=[embedding_matrix],
                input_length=max_sequence_length,
                trainable=False)
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Convolution1D(128, 5)(embedded_sequences)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)
    x = Convolution1D(128, 5)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)
    x = Convolution1D(128, 5)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(class_num, activation='softmax')(x)

    rmsprop = RMSprop(lr=0.001)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])
    return model