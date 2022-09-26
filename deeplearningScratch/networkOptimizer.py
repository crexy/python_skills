import numpy as np

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