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

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 获取概率最高的元素的索引
    if p == np.argmax(t[i]):  # one_hot 标签需用 argmax 取真实类别
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))