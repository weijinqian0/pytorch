# pytorch 实现 线性模型
import torch
from matplotlib import pyplot
from torch.autograd import Variable
import numpy as np
from torch import nn
from torch import optim

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 将narray转化为张量
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# 定义线性模型

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


linear_model = LinearRegression()

# 定义 loss函数(最小二乘) 梯度下降
criterion = nn.MSELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.001)  # 学习率 0.001

# 训练
epochs = 1000
for epoch in range(epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    # 向前传播
    out = linear_model(inputs)
    loss = criterion(out, target)

    # 反向传播
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch + 1) % 20 == 0:
        # print('Epoch[{}/{}], loss: {:6f}'.format(epoch, epochs, loss.data[0]))
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, epochs, loss.data))

linear_model.eval()
predict = linear_model(Variable(x_train))
predict = predict.data.numpy()
pyplot.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
pyplot.plot(x_train.numpy(), predict, label='Fitting Line')
# 显示图例
pyplot.legend()
pyplot.show()

# 保存模型
torch.save(linear_model.state_dict(), './linear.pth')
