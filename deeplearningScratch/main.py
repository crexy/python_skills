from networkLayer import *
from networkOptimizer import *
from multiLayerNetwork import MultiLayerNetwork as mnetwork
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

def overfitting():

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = mnetwork()

    input_node = 784

    layers = []
    layers.append(Affine(np.random.randn(input_node, 300), np.zeros(300)))
    layers.append(Relu())
    layers.append(Affine(np.random.randn(300, 200), np.zeros(200)))
    layers.append(Relu())
    layers.append(Affine(np.random.randn(200, 100), np.zeros(100)))
    layers.append(Relu())
    layers.append(Affine(np.random.randn(100, 50), np.zeros(50)))
    layers.append(Relu())
    layers.append(Affine(np.random.randn(50, 10), np.zeros(10)))
    last_layer = SoftMaxWithLoss()
    network.init_layers(layers, last_layer)
    #network.learning(x_train, t_train, Adam(), 100, 300)

    epoch_times = 200
    batch_size = 100

    optimizer = Adam()

    list_trainAcc = []
    list_testAcc = []

    for ep in range(epoch_times):
        # 미니배치 얻기
        # 전체 데이터가 1000개이고 그중 미니배치 개수가 100 일경우
        # 0~999 사이의 숫자 중 무작위로 중복되지 않는 수 100개를 얻는다.
        mask = DLCFunc.random_index(0, x_train.shape[0], batch_size)  # 중복 없는 미니배치 마스크
        x_batch = x_train[mask]
        t_batch = t_train[mask]

        grad = network.gradient(x_batch, t_batch)
        params = network.params()
        optimizer.update(params, grad)

        acc = network.accuracy(x_train, t_train)
        list_trainAcc.append(acc)

        #mask = DLCFunc.random_index(0, x_test.shape[0], batch_size)  # 중복 없는 미니배치 마스크
        #x_batch = x_test[mask]
        #t_batch = t_test[mask]

        acc = network.accuracy(x_test, t_test)
        list_testAcc.append(acc)

    plt.figure(figsize=(10, 6))
    plt.title("Accuracy value")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")

    plt.plot(list_trainAcc, label = "train")
    plt.plot(list_testAcc, label="test")

    plt.legend()

    plt.show()





if __name__ == "__main__":
    overfitting()

