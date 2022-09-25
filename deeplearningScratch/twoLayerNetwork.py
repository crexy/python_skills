from collections import OrderedDict
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

# 확률적 경사하강법 옵티마이저 클래스
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, param, grad):
        for key in param.keys():
            param[key] -= grad[key]*self.lr

# 모멘텀 옵티마이저 클래스
# GD 기반의 optimizer 최적화 탐색에 v(velocity)개념을 추가하여 GD보다 더 빨리 가중치의 감소, 증가 속도가 빠르고, 가중치의 기울기 방향이 부드럽다.
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, param, grad):
        if self.v == None:
            self.v = {}
            for key, val in param.items():
                self.v[key] = np.zeros_like(val)
        for key in param.keys():
            self.v[key] = self.momentum*self.v[key] - (self.lr*grad[key])
            param[key] += self.v[key]

# 각각의 매개변수에 맞춰서 학습률을 변화시키는 즉 학습을 진행하면서, 학습률을 점차 줄여가는 방법 이다.
# 학습률 감소시키는 기법인 learning rate decay와 비슷하다.
# AdaGrad 알고리즘은 학습률을 감소시키지만 경사가 완만한 차원보다 가파른 차원에 대해 더 빠르게 감소된다.
# 이를 adaptive learning rate라고 부르며 전역 최적점 방향으로 곧장 가도록 갱신하는데 도움이 된다.
# 하지만 Ada Grad 알고리즘은 하습률이 너무 감소되어 전역 최적점에 도착하기 전에 알고리즘이 자주 멈추곤 해서 자주 쓰이지는 않지만,
# 이를 알면 다른 adaptive 학습률 옵티마이저를 이해하는 데 도움이 된다.
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    def update(self, params, grad):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grad[key]*grad[key]
            params[key] -= self.lr*(1/ (np.sqrt(self.h[key])+1e-7))*grad[key]
            #params[key] -= self.lr * grad[key] / (np.sqrt(self.h[key]) + 1e-7)


# Momentum 과 AdaGrad를 융합한 최적화 방법
class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


# AdaGrad에 기울기를 단순 누적하지 않고 지수 가중 이동 평균(EWMA)을 적용한 기법
class RMSprop:
    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Nesterov:
    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""
    # NAG는 모멘텀에서 한 단계 발전한 방법이다. (http://newsight.tistory.com/224)

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


def run_learning(optimizer, network, x_train, t_train, epoch=1000):
    # 학습 구현
    # 배치 사이즈: 100, 에포크: 1000, 최적화: 경사하강법, learning rate: 0.01

    batch_size = 100
    list_loss = []  # 손실값 리스트

    #optimizer = Nesterov()  # 가중치 값 최적화 수행

    for ep in range(epoch):
        mask = np.random.choice(x_train.shape[0], batch_size)  # 배치 마스크
        # 미니배치 데이터
        x_batch = x_train[mask]
        t_batch = t_train[mask]

        # 미분값 구하기
        grad = network.gradient(x_batch, t_batch)

        # 최적화(경사하강법) 수행
        optimizer.update(network.params, grad)

        # 손실값
        loss = network.loss(x_batch, t_batch)
        list_loss.append(loss)  # 손실값 그래프를 그리기 위해 저장
        # 정확도
        accuracy = network.accuracy(x_batch, t_batch)
        if (ep % 200) == 0 and ep != 0:
            print(f'epoch: {ep}, loss: {loss}, accuracy: {accuracy}')
    print(f'epoch: {ep}, loss: {loss}, accuracy: {accuracy}')

    return list_loss

#기울기 검증 하기
def verify_weightGradient(network, x_train, t_train):
    x_batch = x_train[:3]
    t_batch = t_train[:3]
    num_grad = network.numerical_gradient(x_batch, t_batch)
    back_grad = network.gradient(x_batch, t_batch)

    for key in num_grad.keys():
        diff = np.average(num_grad[key] - back_grad[key])
        print(f'Key({key}): {diff}')

# optimizer 성능 비교
def compare_optimizerPerformance(network, x_train, t_train):
    dic_opt = {}
    dic_opt["SGD"] = SGD()
    dic_opt["Monmentum"] = Momentum()
    dic_opt["AdaGrad"] = AdaGrad()
    dic_opt["RMSprop"] = RMSprop()
    dic_opt["Adam"] = Adam()
    dic_opt["Nesterov"] = Nesterov()

    # 손실값 그래프 그리기
    plt.figure(figsize=(10, 6))

    plt.title("Loss value")
    plt.ylabel("loss")
    plt.xlabel("epoch")

    for key in dic_opt.keys():
        print("=============== Optimizer: " + key + " ===============")
        loss_data = run_learning(dic_opt[key], network, x_train, t_train)
        plt.plot(loss_data, label=key)
    plt.legend()
    plt.show()

# 가중치 초기값 성능 비교
def compare_weightInitValuePerformance(x_train, t_train):
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01)
    print("Weight initial value: 0.01")
    dft_std_data = run_learning(SGD(), network, x_train, t_train, epoch=500)

    print("Weight initial value: Xavier")
    Xavier = np.sqrt(1/50.0)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=Xavier)
    xavier_data = run_learning(SGD(), network, x_train, t_train, epoch=500)

    print("Weight initial value: He")
    He = np.sqrt(2/50)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=He)
    he_data = run_learning(SGD(), network, x_train, t_train, epoch=500)

    plt.figure(figsize=(10, 6))
    plt.title("Loss value")
    plt.ylabel("loss")
    plt.xlabel("epoch")

    plt.plot(dft_std_data, label = "std=0.01", marker="o")
    plt.plot(xavier_data, label="Xavier", marker="v")
    plt.plot(he_data, label="He", marker="^")

    plt.legend()
    plt.show()




if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    #network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    # 옵티마이저 성능 비교
    # compare_optimizerPerformance(network, x_train, t_train)

    # 가중치 초기값 성능 비교
    compare_weightInitValuePerformance(x_train, t_train)







