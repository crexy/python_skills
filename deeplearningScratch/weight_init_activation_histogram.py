import numpy as np
import matplotlib.pyplot as plt
from deeplearning_common_func import DLCFunc


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    mask = x <= 0
    x[mask] = 0
    return x

# 은닉층의 활성화 분포 값 확인
def hiddenLayer_activationValue(w_std = 1, node_num=100, activator = sigmoid):
    x = np.random.randn(1000, node_num)  # 1000개의 데이터
    hidden_layer_size = 5  # 은닉층이 5개
    dic_activationVal = {}  # 활성화값을 저장
    dic_w = {}

    for i in range(hidden_layer_size):
        dic_w[i] = np.random.randn(node_num, node_num) * w_std

    for i in range(hidden_layer_size):
        if i != 0:
            x = dic_activationVal[i-1]

        a = np.dot(x, dic_w[i])
        z = activator(a)
        dic_activationVal[i] = z

    for i, val in dic_activationVal.items():
        plt.subplot(1, len(dic_activationVal), i+1)
        plt.title(str(i+1)+"-layer")
        plt.hist(val.flatten(), 30, range=(0,1))
    plt.show()



if __name__ == "__main__":
    node_cnt = 100

    # =============== 활성화 함수: sigmoid ===============

    # 가중치를 표준편차가 1인 정규분로포 초기화
    # 각 층의 활성화 값들이 0과 1에 치우쳐 분포
    # => 시그모이드에서 0과 1은 기울기가 0에 다가감
    # => 역전파의 기울기는 점점 작아지다 사라짐
    # => 기울기 소실문제 발생(층을 깊게하면 기울기 소실이 더 심각)
    # hiddenLayer_activationValue(1)

    # 가중치를 표준편차가 0.01인 정규분로포 초기화
    # 각 층의 활성화 값들이 0.5 부근에 치우쳐 분포
    # => 표현력에 문제, 여러 레이어로 구성되어도 같은 값을 출력 (표현력 제한)
    # => 여러 층을 구성할 필요가 없어짐
    # hiddenLayer_activationValue(0.01)

    # 가중치를 표준편차가 Xavier인 정규분포로 초기화
    # Xavier 값
    # => n: 활성화 계층 이전 계층의 노드 수

    # Xavier = 1 / (np.sqrt(node_cnt))
    # hiddenLayer_activationValue(Xavier, node_cnt)


    # =============== 활성화 함수: Relu ===============

    # std가 0.01일 경우 각층의 활성화 값은 아주 작은 분포로 0에 집중됨
    # 신경망에 아주 작은 분포의 데이터가 흐른다는 것은 역전파 때 가중치의 기울기 또한 작아짐을 의미
    # 실제로 학습이 거의 이루어지지 않음
    # hiddenLayer_activationValue(0.01, node_cnt, relu)

    # Xavier 값 사용: 층이 깊어짐에 따라 분표이 치우침이 커짐
    # Xavier = np.sqrt(1/node_cnt)
    # hiddenLayer_activationValue(Xavier, node_cnt, relu)

    # He 값 사용: 모든 층에서 균일하게 분포
    He = (np.sqrt(2/node_cnt))
    hiddenLayer_activationValue(He, node_cnt, relu)


