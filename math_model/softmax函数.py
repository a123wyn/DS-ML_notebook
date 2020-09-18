'''
softmax classifier for mnist

created on 2019.9.28
author: vince
'''
import math
import logging
import numpy
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


def loss_max_right_class_prob(predictions, y):
    return -predictions[numpy.argmax(y)]


def loss_cross_entropy(predictions, y):
    return -numpy.dot(y, numpy.log(predictions))


'''
Softmax classifier
linear classifier 
'''


class Softmax:

    def __init__(self, iter_num=50000, batch_size=1):
        self.__iter_num = iter_num
        self.__batch_size = batch_size

    def train(self, train_X, train_Y):
        # 加上单位1
        X = numpy.c_[train_X, numpy.ones(train_X.shape[0])]
        Y = train_Y
        self.L = []

        # initialize parameters，m*n矩阵，m为样本特征数，n为标签类别数
        self.__weight = numpy.random.rand(X.shape[1], 4) * 2 - 1.0
        # self.__weight = numpy.zeros((X.shape[1], 4))
        self.__step_len = 1e-4

        logging.info("weight:%s" % (self.__weight))
        for iter_index in range(self.__iter_num):
            if iter_index % 1000 == 0:
                logging.info("-----iter:%s-----" % (iter_index))
            if iter_index % 100 == 0:
                l = 0
                for i in range(0, len(X), 100):
                    predictions = self.forward_pass(X[i])
                    # l += loss_max_right_class_prob(predictions, Y[i]);
                    l += loss_cross_entropy(predictions, Y[i])
                l /= len(X)
                self.L.append(l)
            # 某个数据，梯度下降
            z = numpy.dot(X, self.__weight)
            for i in range(0,z.shape[0]):
                z[i]=z[i]-numpy.max(z[i],axis=0)
            A=numpy.zeros((Y.shape[0],4))
            for i in range(0,len(X)):
                A[i] = numpy.exp(z[i]) / numpy.sum(numpy.exp(z[i]))
            dw = self.__step_len * X.T.dot((A - Y))
            #			dw = self.__step_len * X[sample_index].reshape(-1, 1).dot(predictions.reshape(1, -1));
            #			dw[range(X.shape[1]), numpy.argmax(Y[sample_index])] -= X[sample_index] * self.__step_len;

            self.__weight -= dw
            # sample_index = random.randint(0, len(X) - 1)
            # logging.debug("-----select sample %s-----" % (sample_index))
            #
            # sample_index = random.randint(0, len(X) - 1)
            # logging.debug("-----select sample %s-----" % (sample_index))
            #
            # z = numpy.dot(X[sample_index], self.__weight)
            # z = z - numpy.max(z)
            # predictions = numpy.exp(z) / numpy.sum(numpy.exp(z))
            # dw = self.__step_len * X[sample_index].reshape(-1, 1).dot((predictions - Y[sample_index]).reshape(1, -1))
            # #           dw = self.__step_len * X[sample_index].reshape(-1, 1).dot(predictions.reshape(1, -1));
            # #           dw[range(X.shape[1]), numpy.argmax(Y[sample_index])] -= X[sample_index] * self.__step_len;
            #
            # self.__weight -= dw

            logging.debug("weight:%s" % (self.__weight))
            logging.debug("loss:%s" % (l))
        logging.info("weight:%s" % (self.__weight))
        logging.info("L:%s" % (self.L))
    # softmax
    def forward_pass(self, x):
        net = numpy.dot(x, self.__weight)
        net = net - numpy.max(net)
        net = numpy.exp(net) / numpy.sum(numpy.exp(net))
        return net

    def predict(self, x):
        x = numpy.append(x,1.0)
        return self.forward_pass(x)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    logging.info("trainning begin.")

    # mnist = read_data_sets('../data/MNIST', one_hot=True)  # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签

    # load data
    # train_X = mnist.train.images  # 训练集样本
    # validation_X = mnist.validation.images  # 验证集样本
    # test_X = mnist.test.images  # 测试集样本
    # # labels
    # train_Y = mnist.train.labels  # 训练集标签
    # validation_Y = mnist.validation.labels  # 验证集标签
    # test_Y = mnist.test.labels  # 测试集标签
    train_X = pd.read_excel("xtrain_v2.xlsx", index_col=0)
    train_Y = pd.read_excel("ytrain_v2.xlsx", index_col=0)
    test = pd.read_excel("xtest_v2.xlsx",index_col=0)
    # no = test['id'].values
    # del test['id']
    X_scaler = StandardScaler()
    train_X = X_scaler.fit_transform(train_X)
    test = X_scaler.fit_transform(test)
    # 将目标变量进行labels encoding
    oh = OneHotEncoder()
    train_Y = oh.fit_transform(train_Y)

    train_X,test_X,train_Y,test_Y=train_test_split(train_X,train_Y,test_size=0.1,random_state=0)
    classifier = Softmax()
    classifier.train(train_X, train_Y)

    logging.info("trainning end. predict begin.")

    test_predict = numpy.array([])
    test_right = numpy.array([])
    for i in range(len(test_X)):
        predict_label = numpy.argmax(classifier.predict(test_X[i]))
        test_predict = numpy.append(test_predict, predict_label)
        right_label = numpy.argmax(test_Y[i])
        test_right = numpy.append(test_right, right_label)

    train_predict = numpy.array([])
    train_right = numpy.array([])
    for i in range(len(train_X)):
        train_predict_label = numpy.argmax(classifier.predict(train_X[i]))
        train_predict = numpy.append(train_predict, train_predict_label)
        train_right_label = numpy.argmax(train_Y[i])
        train_right = numpy.append(train_right, train_right_label)

    logging.info("train:right:%s, predict:%s" % (train_right, train_predict))
    train_score = accuracy_score(train_right, train_predict)
    logging.info("The train_accruacy score is: %s " % (str(train_score)))
    logging.info("test:right:%s, predict:%s" % (test_right, test_predict))
    score = accuracy_score(test_right, test_predict)
    logging.info("The test_accruacy score is: %s " % (str(score)))

    output = []
    for i in range(len(test)):
        output.append(classifier.predict(test[i]))
    output = numpy.array(output)
    output = oh.inverse_transform(output)

    d={"grade": list(output)}
    pd_data = pd.DataFrame(d)
    pd_data.to_csv("result.csv")
    # plt.plot(classifier.L)
    # plt.show()

if __name__ == "__main__":
    main()