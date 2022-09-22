from typing import OrderedDict
import numpy as np
from deeplearning_common_func import DLCFunc
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt


# Relu 활성화 함수 레이어
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        # dout 는 배열
        # if dout > 0:
        #     return dout
        # return 0

        # 미분 값이 1이라는 것은 입력 받은것을 그대로 전달한다는 것을 의미, 덧셈 레이어도 입력값이 1을 곱해 그대로 전달해줌
        # 렐루에서 입력 값이 0보다 크면 미분값이 1, 작거나 같으면 0 즉 0보다 크면 값을 그대로 전달
        dout[self.mask] = 0
        dx = dout
        return dx

# Sigmoid 활성화 함수 레이어
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        y = 1/(1+np.exp(-x))
        self.out = y
        return y
    def backward(self, dout):
        out = dout*self.out(1.0-self.out)
        return out

# Affine(행렬의 내적)
class Affine:
    def __init__(self, W, B):
        self.W = W
        self.B = B
        self.dW = None # 가중치의 미분(기울기)
        self.db = None # 편향의 미분
    def forward(self, X):
        self.X = X
        return np.dot(X,self.W)+self.B
    def backward(self, dout):
        # 예 X(2,3), W(3,2), B(2) 형태의 경우
        # 순전파로 결과가(2,2)가됨
        # 역전파의 경우
        # 전달값(L)이 (2,2)임으로 편향(2) 형태로 저장하기 위해서는 sum(axis=0) 수행해야함
        # X의 미분(dX)이 (2,3) 형태를 유지하기 위해서는 (L(2,2) dot W.T(2,3)) 순서로 내적해야함
        # W의 미분(dW)이 (3,2) 현태를 유지하기 위해서는 (X.T(3,2) dot L(2,2)) 순서로 내적해야함
        self.db = np.sum(dout, axis=0)
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        return dX


class SoftMaxWithLoss:
    def __init__(self):
        self.x = None
        self.t = None
        self.y = None
        self.S = None

    def forward(self, x, t):
        self.x = x
        self.t = t
        y = DLCFunc.softmax(x)
        self.y = y
        e = DLCFunc.cross_entropy_error(y, t)
        return e

    def backward(self):
        # L = self.cross_entropy_error_backward()
        # return self.softmax_backward(L)
        # return (self.y - self.t)
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size  # => 왜 batch 사이즈로 나누는지 ????

    def cross_entropy_error_backward(self):
        # L = 1
        # L = -1*1
        # L = L*self.t
        # L = L/self.y
        L = -1 * self.t / self.y
        return L

    def softmax_backward(self, L):
        # L = (-t/y) , y = exp(x)/S, 1/x의 나누기 미분 => -1/x**2
        # bs = L/self.S
        # bs = bs*(-1/self.S**2) # => 1/self.S
        # ba = L*(1/self.S)
        # bl = (bs+ba)*np.exp(self.x)
        bl = self.y - self.t
        return bl


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치, 편향 행렬 초기화
        self.params = {}
        self.params["W1"] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params["b2"] = np.zeros(output_size)

        # 신경망 레이어 초기화(입력/은닉층=>활성화층=>은닉/출력증)
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        # 손실함수 레이어
        self.lastLayer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        e = self.lastLayer.forward(y, t)
        return e

    def accuracy(self, x, t):
        y = self.predict(x)
        tidx = t.copy()
        if tidx.ndim == 2:
            tidx = np.argmax(tidx, axis=1)
        ymax_idx = np.argmax(y, axis=1)
        match_cnt = np.sum(ymax_idx == tidx)
        return match_cnt / float(x.shape[0])

    def numerical_gradient(self, x, t):  # 역전파 gradient 함수 검증 함수
        loss_W = lambda: self.loss(x, t)

        grads = {}
        grads['W1'] = DLCFunc.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = DLCFunc.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = DLCFunc.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = DLCFunc.numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):

        # 순전파
        self.loss(x, t)

        dout = self.lastLayer.backward()

        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    # 기울기 검증 하기
    #
    # x_batch = x_train[:3]
    # t_batch = t_train[:3]
    #
    # num_grad = network.numerical_gradient(x_batch, t_batch)
    # back_grad = network.gradient(x_batch, t_batch)
    #
    # for key in num_grad.keys():
    #     diff = np.average(num_grad[key] - back_grad[key])
    #     print(f'Key({key}): {diff}')

    # 학습 구현
    # 배치 사이즈: 100, 에포크: 1000, 학습방법: 경사하강법, learning rate: 0.01

    batch_size = 100
    epoch = 1000
    learning_rate = 1e-1
    list_loss = [] # 손실값 리스트

    for ep in range(epoch):
        mask = np.random.choice(x_train.shape[0], batch_size) # 배치 마스크
        # 미니배치 데이터
        x_batch = x_train[mask]
        t_batch = t_train[mask]

        # 미분값 구하기
        grad = network.gradient(x_batch, t_batch)

        # 경사하강법 학습 수행
        for key in grad.keys():
            network.params[key] -= grad[key]*learning_rate

        # 손실값
        loss = network.loss(x_batch, t_batch)
        list_loss.append(loss) # 손실값 그래프를 그리기 위해 저장
        # 정확도
        accuracy = network.accuracy(x_batch, t_batch)
        if (ep % 10) == 0:
            print(f'epoch: {ep+1}, loss: {loss}, accuracy: {accuracy}')
    # 손실값 그래프 그리기
    plt.figure(figsize=(10,6))
    plt.title("Loss value")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.plot(list_loss)
    plt.show()





