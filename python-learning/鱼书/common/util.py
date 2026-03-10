# coding: utf-8
"""
通用工具函数模块
包含：曲线平滑、数据集打乱、卷积尺寸计算、im2col/col2im 转换等
"""
import numpy as np


def smooth_curve(x):
    """对损失函数曲线进行平滑处理，使训练过程中的波动更易于观察

    使用 Kaiser 窗口进行卷积，可有效减少噪声，得到更平滑的曲线。
    常用于绘制训练 loss 曲线时，避免因 batch 波动导致的锯齿状图形。

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html

    Parameters
    ----------
    x : array-like
        原始数据序列，通常为每个 epoch 或 step 的 loss 值

    Returns
    -------
    y : ndarray
        平滑后的数据序列，长度略短于输入
    """
    window_len = 11
    # 在序列两端做镜像扩展，避免卷积时边界失真
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # 使用 Kaiser 窗口作为卷积核
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    # 去掉因扩展产生的两端多余部分
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """对数据集进行随机打乱，打乱样本顺序的同时保持特征与标签的对应关系

    常用于训练前或每个 epoch 开始时，避免模型学习到数据顺序带来的偏差，
    提高泛化能力。

    Parameters
    ----------
    x : ndarray
        训练数据，形状为 (样本数, 特征...) 或 (样本数, 通道, 高, 宽)
    t : ndarray
        标签数据，形状为 (样本数,) 或 (样本数, 类别数)

    Returns
    -------
    x : ndarray
        打乱后的训练数据
    t : ndarray
        打乱后的标签数据，与 x 的样本一一对应
    """
    # 生成随机排列的索引，保证 x 和 t 使用相同的打乱顺序
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    """计算卷积操作后输出特征图在某一维度上的尺寸

    公式：output_size = (input_size + 2*pad - filter_size) / stride + 1

    Parameters
    ----------
    input_size : int
        输入在该维度上的尺寸（高度或宽度）
    filter_size : int
        卷积核在该维度上的尺寸
    stride : int, optional
        步长，默认为 1
    pad : int, optional
        填充像素数，默认为 0

    Returns
    -------
    float
        输出特征图在该维度上的尺寸
    """
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """将输入图像展开为列矩阵（im2col），便于用矩阵乘法高效实现卷积

    卷积层的前向传播中，将每个卷积核滑窗内的像素展平成一列，
    所有滑窗组成一个大矩阵，可与卷积核权重做一次矩阵乘法完成卷积计算。
    这是将卷积运算转化为矩阵运算的经典优化方法。

    Parameters
    ----------
    input_data : ndarray
        输入数据，4 维数组，形状为 (批大小, 通道数, 高度, 宽度)
    filter_h : int
        卷积核的高度
    filter_w : int
        卷积核的宽度
    stride : int, optional
        滑动步长，默认为 1
    pad : int, optional
        四周填充的像素数，默认为 0

    Returns
    -------
    col : ndarray
        2 维数组，每行对应一个滑窗内展平的像素，用于与卷积核做矩阵乘法
    """
    N, C, H, W = input_data.shape
    # 计算卷积后输出特征图的高度和宽度
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # 对输入进行 padding，便于处理边界
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 按滑窗位置提取像素，每个 (y, x) 对应卷积核内的一个位置
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 调整维度顺序并展平：将每个滑窗展成一行
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """将列矩阵还原为图像格式（col2im），是 im2col 的逆操作

    在卷积层的反向传播中，梯度是以 im2col 展开后的列形式存在的，
    需要将其还原为原始的 4 维图像形状，以便传递给上一层。
    同一位置的像素可能被多个滑窗覆盖，还原时需要将梯度累加。

    Parameters
    ----------
    col : ndarray
        im2col 展开后的 2 维数组，由 im2col 或卷积前向传播产生
    input_shape : tuple
        原始输入数据的形状，例如 (批大小, 通道数, 高度, 宽度)，如 (10, 1, 28, 28)
    filter_h : int
        卷积核的高度，需与 im2col 时使用的参数一致
    filter_w : int
        卷积核的宽度，需与 im2col 时使用的参数一致
    stride : int, optional
        滑动步长，默认为 1
    pad : int, optional
        四周填充的像素数，默认为 0

    Returns
    -------
    img : ndarray
        还原后的 4 维数组，形状为 (批大小, 通道数, 高度, 宽度)
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # 将 col 从 2 维还原为 6 维，对应 im2col 的中间格式
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 初始化输出图像，将每个滑窗的梯度填回对应位置（使用 += 累加重叠部分）
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 去掉 padding 部分，返回原始尺寸
    return img[:, :, pad:H + pad, pad:W + pad]