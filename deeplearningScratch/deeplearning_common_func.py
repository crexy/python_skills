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

    @staticmethod
    def random_index(start, end, size):
        arr = np.arange(start, end)
        np.random.shuffle(arr)
        return arr[:size]


if __name__ == "__main__":
    x = np.random.uniform(low=0, high=1, size=(2,3))
    # print(x)
    # print("self")
    # print(DLCFunc.softmax(x))
    # print("ref")
    # print(softmax(x))

    DLCFunc.numerical_gradient()




    