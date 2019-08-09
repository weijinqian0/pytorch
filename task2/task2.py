import numpy as np
import torch
from torch.autograd import Variable

"""
【Task2(2天)】设立计算图并自动计算
1.numpy和pytorch实现梯度下降法
    a.设定初始值
    b.求取梯度
    c.在梯度方向上进行参数的更新
2.numpy和pytorch实现线性回归
3.pytorch实现一个简单的神经网络
4.参考资料：PyTorch 中文文档 https://pytorch.apachecn.org/docs/1.0/

"""


def numpyGradient(init_x, epochs, gradient, learning_rate=0.1):
    """
    用numpy学习梯度下降法
    :param init_x: 初始x
    :param epochs: 迭代次数
    :param gradient: 梯度表达式
    :param learning_rate: 学习系数
    :return:
    """
    x = init_x
    for epoch in range(epochs):
        x -= learning_rate * gradient(x)

    print(x)


def pytorchGradient(init_x, epochs, gradient, learning_rate=0.1):
    """
    使用pytorch学习梯度下降法
    :param init_x:
    :param epochs:
    :param gradient:
    :param learning_rate:
    :return:
    """
    x = init_x
    y = gradient(x)
    parameters = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    for epoch in range(epochs):
        y.backward(parameters, retain_graph=True)
        x.data = x.data - learning_rate * x.grad.data
        x.grad.data.zero_()

    print(x)





if __name__ == '__main__':
    # 假设f(x)=x**2+2*x+1 最小值为x=-1
    # a.设定初始值
    x = np.random.rand(3, 2)
    print(x)
    numpyGradient(x, 1000, lambda x: 2 * x + 2)

    x = torch.rand(3, 2, requires_grad=True)
    print(x)
    pytorchGradient(x, 10, lambda x: x ** 2 + 2 * x + 1)
