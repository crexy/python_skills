import numpy as np
from deeplearning_common_func import DLCFunc

class MultiLayerNetwork:
    def __init__(self, weight_init_std = "H"):
        self.layers = None
        self.lastLayer = None
        self.weight_init_std = weight_init_std

    def init_layers(self, _layers, _lastLayer):

        for i, layer in enumerate(_layers):
            # 마지막 레이어가 아닌 경우
            if i < len(_layers)-1:
                # 현재 레이어가 활성화 층인 경우
                if layer.__class__.__name__ == "Relu" or \
                    layer.__class__.__name__ == "Sigmoid":
                    # 다음 은닉층의 노드 갯수
                    node_cnt = _layers[i+1].W.shape[0]
                    init_std = 0
                    if self.weight_init_std == "H": # 가중치 표준편차 초기값 Xavier
                        init_std = np.sqrt(1/node_cnt)
                    elif self.weight_init_std == "X": # 가중치 표준편차 초기값 He
                        init_std = np.sqrt(2/node_cnt)
                    _layers[i + 1].W *= init_std
        self.layers = _layers
        self.lastLayer = _lastLayer

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def accuracy(self, x, t):
        y = self.predict(x)
        # 정답 라벨의 형태확인
        if x.ndim == t.ndim: # one hot
            # 정답 레벨의 형태를 인덱스 정답으로 변경
            t = np.argmax(t, axis=1).copy()
        y = np.argmax(y, axis=1)
        return np.sum(y == t)/t.shape[0]

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # 순전파 =>
        # 순전파를 통해 은닉층에 x 를 저장시킴
        # 순전파를 통해 마지막 층에 x와 t를 저장 시킴
        # 저장된 x, t 데이터는 역전파 시 미분값을 얻는데 사용됨
        L = self.loss(x, t)
        dout = self.lastLayer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        grad = {}
        for i, layer in enumerate(self.layers):
            if layer.__class__.__name__ == "Relu" or \
                    layer.__class__.__name__ == "Sigmoid":
                continue
            key = "W"+str(i)
            grad[key] = layer.dW
            key = "b" + str(i)
            grad[key] = layer.db
        return grad

    def params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if layer.__class__.__name__ == "Relu" or \
                    layer.__class__.__name__ == "Sigmoid":
                continue
            key = "W" + str(i)
            params[key] = layer.W
            key = "b" + str(i)
            params[key] = layer.b
        return params

    def learning(self, x, t, optimizer, batch_size, epoch_times):
        for ep in range(epoch_times):
            # 미니배치 얻기
            # 전체 데이터가 1000개이고 그중 미니배치 개수가 100 일경우
            # 0~999 사이의 숫자 중 무작위로 중복되지 않는 수 100개를 얻는다.
            mask = DLCFunc.random_index(0, x.shape[0], batch_size) # 중복 없는 미니배치 마스크
            x_batch = x[mask]
            t_batch = t[mask]

            grad = self.gradient(x_batch, t_batch)
            params = self.params()

            optimizer.update(params, grad)

        loss = self.loss(x_batch, t_batch)
        accracy = self.accuracy(x_batch, t_batch)

        print(f'Accuracy: {accracy}, Loss: {loss}')

