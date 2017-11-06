#coding = utf-8

import numpy as np
import matplotlib.pyplot as plt

cost = []
t = []

class BPNet:
    def __init__(self, input_num, hidden_num, output_num, learning_rate):
        self.input = np.zeros((1, input_num))
        self.W0 = 2 * np.random.random_sample((input_num, hidden_num)) - 1.0
        self.W1 = 2 * np.random.random_sample((hidden_num, output_num)) - 1.0
        self.b0 = np.random.random_sample((1, hidden_num))
        self.b1 = np.random.random_sample((1, output_num))
        self.lr = learning_rate
        self.L0 = 0
        self.L1 = 0

    def forward(self, Input):
        self.input = Input
        self.L0 = self.sigmod(np.dot(Input, self.W0) + self.b0)
        self.L1 = self.sigmod(np.dot(self.L0, self.W1) + self.b1)
        return self.L1

    def backward(self, loss):
        l1_delta = loss * self.sigmod(self.L1, True)
        l0_error = np.dot(l1_delta, self.W1.T)
        l0_delta = l0_error * self.sigmod(self.L0, True)
        self.W1 += self.lr * np.dot(self.L0.T, l1_delta)
        self.W0 += self.lr * np.dot(self.input.T, l0_delta)

    def train(self, train_x, train_y, train_num, batch_size = 10):
        for i in range(train_num):
            batch_num = int(len(train_y) / batch_size)
            for j in range(batch_num):
                batch_y = train_y[j * batch_num: batch_num]
                batch_x = train_x[j * batch_num: batch_num]
                result = self.forward(batch_x)
                loss = batch_y - result
                self.backward(loss)
            if i % 100 == 0:
                preY = self.forward(train_x)
                mseError = 0.5 * np.sum(np.mean((preY - train_y) * (preY - train_y)))
                cost.append(mseError)
                t.append(i)
                print("epoch {}: loss = {}".format(i, mseError))

    def predict(self, test_x):
        res = self.forward(test_x)
        res = np.array(res).argmax(1)
        return res

    def sigmod(self, x, div = False):
        if div is True:
            return x * (1 - x)
        else:
            return 1/(1 + np.exp(-x))


def loadData(filepath):
    feature = []
    label = []
    with open(file=filepath) as f:
        line = f.readline()
        while line:
            x1, x2, x3, x4, y = line.rstrip().split(',')
            feature.append([float(x1), float(x2), float(x3), float(x4)])
            label.append([float(y)])
            line = f.readline()
    return feature, label


def singleToOneHot(label, label_num):
    onehot = np.zeros((len(label), label_num))
    for i in range(len(label)):
        pos = int(label[i][0])
        onehot[i][pos - 1] = 1.0
    return onehot



if __name__ == '__main__':
    X, Y = loadData('data.txt')
    testX, testY = loadData('test.txt')
    X = np.array(X)
    Y = np.array(singleToOneHot(Y, 3))
    net = BPNet(input_num=4, hidden_num=10, output_num=3, learning_rate=0.1)
    net.train(train_x=X, train_y=Y, train_num=10000, batch_size=1)
    predictVal = net.predict(test_x=testX)
    print(predictVal)
    plt.plot(t, cost)
    plt.show()
