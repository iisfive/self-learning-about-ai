# 本片段用于展示所下载的MNIST数据集的图像
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

# 指定 MNIST 数据集路径（你的数据在 raw 子目录下）
MNIST_PATH = r"D:\postgraduate\AI_learning\self_learning\python-learning\爆肝杰哥-深度学习系列讲义\chapter5-深度神经网络（DNN）-本地\mnist\MNIST\raw"


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))    # 需要把保存为NumPy数组的图像数据转换成PIL用的数据对象
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten=True, normalize=False, dataset_dir=MNIST_PATH
)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,) 原因是:flatten=True
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)