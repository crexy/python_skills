import numpy as np

class DLCFunc: # deeplearning common function
    def __init__(self):
        pass
    @staticmethod
    def softmax(x):        
        if x.ndim == 2:
            '''
            ex) x가 (2,3) 행렬일 경우
            각 행의 최대값을 구하면 (2,) 행렬이 생성되는데 
            (2,3) - (2,) 은 행렬 연산이 불가함으로
            x 행렬을 역행렬로 변환해 최대값을 차감해줘야 한다.
            x.T =>(3,2) 
            (3,2) - (2,) 으로 최대값을 차감하고 소프트 맥스 연산을 수행한 뒤
            결과 값에 역행렬 변환을 해준다.
            '''
            x = x.T 
            # 오버플로 대책으로 최대값 차감, exp()지수함수는 e^n의 값으로 n의 값이 커지면 결과가 inf가 나오는 오버플로 발생
            max = np.max(x, axis=0)       
            x = x-max     
            sum = np.sum(np.exp(x), axis=0)
            y = (np.exp(x)/sum).T
        else:    
            max = np.max(x)
            x = x-max
            sum = np.sum(np.exp(x))
            y = np.exp(x)/sum
        return y

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def cross_entropy_error(cls, y, t):
        '''
        정답이 원핫레이블 형태인지 아니면 정답 인덱스만 있는지 여부를 확인하여 오차계산 수행
        '''
        if y.ndim == t.ndim: # 원핫레이블 형태
            return -np.sum(t*np.log(y))/t.shape[0]
        
        # 인덱스 정답인경우
        return -np.sum(np.log(y[range(y.shape[0]), t]))/t.shape[0]

    @staticmethod
    def numerical_gradient(f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h # f(x+h)
            fxh1 = f()  

            x[idx] = tmp_val - h # f(x-h)
            fxh2 = f()  
            grad[idx] = (fxh1 - fxh2) / (2*h)

            x[idx] = tmp_val  # 값 복원
            it.iternext()

        return grad        

# def softmax(x):
#     if x.ndim == 2:
#         x = x.T
#         x = x - np.max(x, axis=0)
#         y = np.exp(x) / np.sum(np.exp(x), axis=0)
#         return y.T 

#     x = x - np.max(x) # 오버플로 대책
#     return np.exp(x) / np.sum(np.exp(x))


# def mean_squared_error(y, t):
#     return 0.5 * np.sum((y-t)**2)
        
# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
        
#     # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
#     if t.size == y.size:
#         t = t.argmax(axis=1)
             
#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

if __name__ == "__main__":
    x = np.random.uniform(low=0, high=1, size=(2,3))
    # print(x)
    # print("self")
    # print(DLCFunc.softmax(x))
    # print("ref")
    # print(softmax(x))

    DLCFunc.numerical_gradient()




    