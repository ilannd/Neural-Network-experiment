from __future__ import division
import math
import random
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

flowerLables = {0: 'Iris-setosa',
                1: 'Iris-versicolor',
                2: 'Iris-virginica'}

random.seed(0)  # 只要我们设置相同的seed,就能确保每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数


# 生成区间[a, b)内的随机数
def rand(a, b):
    return (b - a) * random.random() + a  # random.random() 生成0—1之间的随机数


# 生成大小 I*J 的矩阵，默认零矩阵
def makeMatrix(I, J, fill=0.0):  # fill = 0.0为默认值参数，调用函数可修改默认值
    m = []
    for i in range(I):
        m.append([fill] * J)  # m.append列表尾部追加成员; [0.0]是一个列表，列表支持乘法运算，[0.0]*5 的结果是[0.0 0.0 0.0 0.0 0.0 0.0]
    return m  # 在列表尾部追加成员也是列表，所以返回的是个矩阵


# 函数 sigmoid，bp神经网络前向传播的激活函数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# 函数 sigmoid 的导数,反向传播时使用
def dsigmoid(x):
    return x * (1 - x)


errors = []


class NN:
    """ 三层反向传播神经网络 """

    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1  # 输入层增加一个偏差节点。定义一个实例属性self.ni记录输入节点个数
        self.nh = nh + 1  # 隐藏层增加一个偏置节点
        self.no = no  # 输出层个数即为鸢尾花的种类数

        # 激活神经网络的所有节点（向量）
        self.ai = [1.0] * self.ni  # 输入层神经元的激活项其实就是输入的特征数，这样设计是为了向量化前向传播过程。
        self.ah = [
                      1.0] * self.nh  # [1.0]是一个列表，列表支持乘法运算，[1.0]*5 的结果是[1.0 1.0 1.0 1.0 1.0 1.0]。偏置节点必定为1，也就是列表中第一个数为1，其他数依次记录隐藏层节点的激活项
        self.ao = [1.0] * self.no  # 输出层输出的结果为预测该样本属于某一类的概率，概率最大者，则预测为该类

        # 建立权重（矩阵）
        self.wi = makeMatrix(self.ni, self.nh)  # 定义一个实例属性self.wi记录输入层到隐藏层的映射矩阵
        self.wo = makeMatrix(self.nh, self.no)  # python中类中可以直接调用全局函数makeMatrix()生成矩阵，默认零矩阵；隐藏层到输出层的映射矩阵。
        # 设为随机值，在做线性回归和逻辑回归时，一般将权重都初始化为0；但在神经网络中需要设为随机值，是因为如果权重矩阵为零矩阵的话，那么经过前向传播下一层神经元的激活项均相同
        for i in range(self.ni):  # 上层循环控制行
            for j in range(self.nh):  # 下层循环控制列
                self.wi[i][j] = rand(-0.2, 0.2)  # 调用全局函数rand(),生成区间[-0.2, 0.2)内的随机数
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2, 2)  # 生成区间[-2, 2)内的随机数

    '''前向传播，激活神经网络的所有节点（向量)'''

    def update(self, inputs):
        if len(inputs) != self.ni - 1:  # 输入的样本特征量数等于神经网络输入层数-1，因为有一个是偏置节点
            raise ValueError('与输入层节点数不符！')  # 使用raise手工抛出异常，若引发该异常，中断程序

        # 激活输入层
        for i in range(self.ni - 1):  # 输入层中的偏置节点 = 1，不用激活
            self.ai[i] = inputs[i]  # 将输入样本的特征量赋值给神经网络输入层的其他节点

        # 激活隐藏层
        for j in range(self.nh):  # self.nh表示隐藏层的节点数，包括隐藏层的第一个节点，也就是我们人为加的偏置节点，偏置节点恒为1，是不需要激活的；应该是self.nh -1,但原代码也并不影响结果
            sum = 0.0  # 激活项a = g(z)  z = Θ^T x ;sum相当于z，每次循环归零
            for i in range(self.ni):  # 通过循环z = Θ^T x ，因为Θ、x均为向量
                sum = sum + self.ai[i] * self.wi[i][j]  # 〖 Z〗^((2))=Θ^((1)) a^((1))
            self.ah[j] = sigmoid(sum)  # a^((2))=g(z^((2)))，这里使用sigmoid()函数作为激活函数

        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]  # 〖 Z〗^((3))=Θ^((2)) a^((2))
            self.ao[k] = sigmoid(sum)  # a^((3))=g(z^((3)))

        return self.ao[:]  # 返回输出值，即为某样本的预测值

    '''反向传播, 计算节点激活项的误差'''

    def backPropagate(self, targets, lr):  # targets为某样本实际种类分类，lr为梯度下降算法的学习率

        # 计算输出层的误差
        output_deltas = [
                            0.0] * self.no  # 记录方向传播的误差；输出层误差容易求，把样本的实际值减去我们当前神经网络预测的值，δ^((3))=〖y-a〗^((3) );但是输出层的误差是由前面层一层一层累加的结果，我们将误差方向传播的过程叫方向传播算法。由算法知：δ^((2))=〖(Θ^((2)))〗^T δ^((3)).*g^' (z^((2)))
        for k in range(self.no):
            error = targets[k] - self.ao[k]  # δ^((3))=〖y-a〗^((3) ),得到输出层的误差
            output_deltas[k] = dsigmoid(
                self.ao[k]) * error  # dsigmoid()函数的功能是求公式中 g^' (z^((2))) 项，而output_deltas记录的是δ^((3)).*g^' (z^((2)))的值

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh  # 记录的是δ^((2)).*g^' (z^((1)))的值
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]  # 求δ^((2))，隐藏层的误差
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # 更新输出层权重
        for j in range(
                self.nh):  # 反向传播算法，求出每个节点的误差后，反向更新权重；由算法知Δ(_ij ^((L)))=Δ(_ij ^((L)))+a(_j  ^((L)))δ(_i      ^((L+1)))    ,而∂/(∂Θ_ij^((L) ) ) J(Θ)=Δ_ij^((L))   (λ=0) λ为正则化系数。代入梯度下降算法中：Θ_ij^((L))=Θ_ij^((L))+α  ∂/(∂Θ_ij^((L) ) ) J(Θ)即可更新权重
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]  # 求 a(_j  ^((L)))δ(_i      ^((L+1)))  项
                self.wo[j][k] = self.wo[j][k] + lr * change  # 用于梯度下降算法

        # 更新输入层权重
        for i in range(self.ni):  # 与上同理
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + lr * change

        # 计算误差
        error = 0.0  # 每调用一次先归零，不停地进行迭代
        error += 0.5 * (targets[k] - self.ao[k]) ** 2  # 神经网络的性能度量，其实就是均方误差少了除以整数，但不影响度量
        return error  # 返回此时训练集的误差

    # 用测试集来测试训练过后的神经网络，输出准确率
    def test(self, patterns):  # patterns为测试样本数据
        count = 0  # 记录测试样本的实际值与预测值相等的个数
        for p in patterns:
            target = flowerLables[
                (p[1].index(1))]  # p[1].index(1)：返回p[1]列表中值为1的序号；而这序号正对应flowerLables字典中的键值。target存储的是样本实际种类类别
            result = self.update(p[0])  # 输入测试样本的特征值，返回的是对每种种类预测的概率
            index = result.index(max(result))  # 求出result列表中最大数值的序号
            print(p[0], ':', target, '->', flowerLables[index])  # 输出测试样本的特征值，实际输出，预测输出
            count += (target == flowerLables[
                index])  # 若样本的实际值与预测值相等为真，加1。顺便提一下，其实flowerLables字典完全没有必要使用，我们只要确定p[1].index(1)与index相等即可
        accuracy = float(count / len(patterns))  # 求准确率
        print('accuracy: %-.9f' % accuracy)

    # 输出训练过后神经网络的权重矩阵
    def weights(self):
        print('输入层权重:')
        for i in range(self.ni):
            print()
        print('输出层权重:')
        for j in range(self.nh):
            print(self.wo[j])

    # 用训练集训练神经网络
    def train(self, patterns, iterations=1000,
              lr=0.1):  # patterns:训练集数据 iterations:迭代次数，默认值为1000；lr: 梯度下降算法中的学习速率(learning rate）
        for i in range(iterations):  # 这里默认规定了梯度下降算法迭代的次数
            error = 0.0  # 记录每次迭代后的误差
            for p in patterns:  # 将训练集的数据依次喂入神经网络输入层
                inputs = p[0]  # inputs获取该样本的特征值
                targets = p[1]  # targets获取该样本的种类类别
                self.update(inputs)  # 前向传播，激活神经网络每个节点
                error = error + self.backPropagate(targets, lr)  # 反向传播，算出每个节点的误差，并通过反向传播算法更新权重，算出此时的样本误差
                errors.append(error)
            if i % 100 == 0:  # 方便我们观看样本误差变化情况
                print('error: %-.9f' % error)

        return errors


def iris():
    data = []  # 建立一个data列表，用来存放样本数据
    # 读取数据
    raw = pd.read_csv('iris.csv')  # pd是pandas模块的重命名，pd.read_cs()函数读取本地文件iris.csv里数据。raw为DataFrame类型
    raw_data = raw.values  # 将DataFrame类型转化为array类型
    raw_feature = raw_data[0:, 0:4]  # 用冒号表达式取数，取第1-5列的数，也就是样本的特征值
    for i in range(len(raw_feature)):  # 将数据保存在列表中，方便后面操作
        ele = []
        ele.append(list(raw_feature[i]))  # ele列表第一个元素保存该样本特征值
        if raw_data[i][4] == 'Iris-setosa':  # 用向量表示种类类型，Iris-setosa用[1,0,0]表示
            ele.append([1, 0, 0])  # ele列表第二个元素该样本的种类
        elif raw_data[i][4] == 'Iris-versicolor':  # Iris-versicolor用[0,1,0]表示
            ele.append([0, 1, 0])
        else:
            ele.append([0, 0, 1])  # Iris-virginica用[0,0,1]表示
        data.append(ele)  # 将ele列表作为一个元素加入到data列表中
        '''ele1=np.array(ele)
        cha=ele1.shape[0]
        y1 = []
        for i in range(cha):
            if ele[i][0] == 1:
                y1.append(i)
        y2 = []
        for i in range(cha):
            if ele[i][1] == 1:
                y2.append(i)
        y3 = []
        for i in range(cha):
            if ele[i][2] == 1:
                y3.append(i)
'''
    # 随机排列数据
    random.shuffle(data)  # 将样本次序随机打乱
    training = data[0:100]  # 取序号0-100作为训练集
    test = data[101:]  # 取序号100-150作为测试集
    nn = NN(4, 7, 3)  # 用nn实例化NN类，同时调用构造函数建立bp神经网络结构
    nn.train(training, iterations=200)  # 调用类中方法train(),通过反向传播算法训练权重
    nn.test(test)  # 调用类中方法test(),用测试集来测试训练过后的神经网络
    x = np.arange(0, (len(errors)))
    plt.figure()
    plt.plot(x, errors)
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.show()


# 当.py文件被直接运行时，iris()函数被运行；当.py文件以模块形式被导入时，iris()函数不被运行。可以认为这是程序的入口
if __name__ == '__main__':
    iris()
