# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    '''
    一维数值梯度计算
    '''
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)    # 初始化梯度为0, x 的形状为(N,)

    for idx in range(x.size):
        tmp_val = x[idx]    # 保存当前值
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val    # 恢复当前值
        
    return grad

def numerical_gradient_2d(f, x):
    '''
    二维数值梯度计算
    '''
    if x.ndim == 1:
        return _numerical_gradient_1d(f, x)  # 如果x是一维数组，则调用一维数值梯度计算
    else:
        grad = np.zeros_like(x)  # 初始化梯度为0, x 的形状为(N, D)
        for idx in range(x.shape[0]):
            grad[idx] = _numerical_gradient_1d(f, x[idx])  # 对每一维调用一维数值梯度计算
        return grad

def numerical_gradient(f, x):
    '''
    数值梯度计算
    '''
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    # 使用nditer遍历x的每个元素，nditer是numpy库中的一个函数，用于遍历数组中的每个元素
    # flags=['multi_index']表示使用多索引，op_flags=['readwrite']表示读写模式
    # 举一个例子说明什么是数组x的多索引：
    # x = np.array([[1, 2], [3, 4]])，那么x.shape = (2, 2)，
    # 那么x的多索引就是(0, 0), (0, 1), (1, 0), (1, 1)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        it.iternext()
    return grad
