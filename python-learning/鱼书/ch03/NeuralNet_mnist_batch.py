# 当前脚本仅作测试使用，未进行训练
import sys, os
# 将项目根目录（鱼书）加入路径，确保能导入 common、dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from common.functions import sigmoid, softmax

MNIST_PATH = r"D:\postgraduate\AI_learning\self_learning\python-learning\爆肝杰哥-深度学习系列讲义\chapter5-深度神经网络（DNN）-本地\mnist\MNIST\raw"
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True, dataset_dir=MNIST_PATH)
    return x_test, t_test

def init_network():
    '''
    从pickle文件中加载权重和偏置参数,并返回一个包含权重和偏置的字典变量network
    '''
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    '''
    预测函数,根据输入的网络和输入数据,返回预测结果
    '''
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()    # 获取测试数据
network = init_network()

batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):

    # 从输入数据中抽出批数据,x[i:i+batch_n] 会取出从第 i 个到第 i+batch_n 个之间的数据, 从头开始以 batch_size 为单位将数据提取为批数据
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)

    # 获取预测结果中概率最高的元素的索引，axis=1 表示按行取最大值,即以第1维为轴，找每一行中的最大值
    p_batch = np.argmax(y_batch, axis=1)

    # 获取真实标签中概率最高的元素的索引
    t_batch = np.argmax(t[i:i+batch_size], axis=1)
    accuracy_cnt += np.sum(p_batch == t_batch)
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
