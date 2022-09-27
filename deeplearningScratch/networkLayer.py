import numpy as np
from deeplearning_common_func import DLCFunc

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
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dW = None # 가중치의 미분(기울기)
        self.db = None # 편향의 미분
    def forward(self, X):
        self.X = X
        return np.dot(X,self.W)+self.b
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


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            #x[self.mask] = 0
            x = x * self.mask
        else:
            x = x * (1-self.dropout_ratio)
        return x
    def backward(self, dout):
        return self.mask * dout