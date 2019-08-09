import numpy as np


def weightsUpdate(data, w, b, learning_rate=0.01):
    for x0, y0 in data:
        y = np.dot(x0, w) + b
        print("x0", x0)
        print("y0",y0)
        w_gradient = (y - y0) * x0.T  # 求w的梯度
        b_gradient = (y - y0)[0]  # 求b的梯度；取标量
        w -= w_gradient * learning_rate  # 权值更新
        b -= b_gradient
        loss = 0.5 * np.square(y - y0)
    return [w, b, loss[0][0]]


def generateData(w, b, dataNum=10):  # 根据w,b生成数据
    data = []

    for i in range(dataNum):
        noise = np.random.randn(1) * 0.01
        x0 = np.random.randn(1, w.shape[0])
        y0 = np.dot(x0, w) + b + noise
        x = [x0, y0]
        data.append(x)
    return data


def linearRegressionTrain(data):
    w0 = np.random.randn(data[0][0].shape[1], 1)
    b0 = np.random.randn(1)
    for i in range(1000):
        w0, b0, loss = weightsUpdate(data, w0, b0, 0.01)
        if (i % 100 == 0):
            print(loss)

    return [w0, b0]


# y=2*x1+3*x2+1
w = np.array([[2], [3], [4], [5]])
b = np.array([1])

data = generateData(w, b)
print(data)
w0, b0 = linearRegressionTrain(data)

print(" w=", w, '\n', "w0=", w0, '\n', "b=", b, '\n', "b0=", b0)
