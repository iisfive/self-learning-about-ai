# coding: utf-8
"""
鱼书 MNIST 数据集加载模块
支持自定义数据集路径，兼容 .gz 压缩文件和已解压的 .ubyte 文件
"""
import os
import gzip
import pickle
import numpy as np

# 默认路径：与 mnist.py 同目录
_default_dir = os.path.dirname(os.path.abspath(__file__))

key_file = {
    'train_img': 'train-images-idx3-ubyte',
    'train_label': 'train-labels-idx1-ubyte',
    'test_img': 't10k-images-idx3-ubyte',
    'test_label': 't10k-labels-idx1-ubyte',
}

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _get_file_path(dataset_dir, file_name):
    """获取文件路径，支持 raw 子目录（PyTorch 风格）"""
    base_path = os.path.join(dataset_dir, file_name)
    raw_path = os.path.join(dataset_dir, 'raw', file_name)
    gz_path = base_path + '.gz'
    raw_gz_path = raw_path + '.gz'
    
    # 优先查找已解压文件
    if os.path.exists(base_path):
        return base_path, False  # False 表示非压缩
    if os.path.exists(raw_path):
        return raw_path, False
    if os.path.exists(gz_path):
        return gz_path, True  # True 表示 gzip 压缩
    if os.path.exists(raw_gz_path):
        return raw_gz_path, True
    return None, None


def _load_label(file_path, is_gzip):
    """加载标签文件"""
    open_fn = gzip.open if is_gzip else open
    mode = 'rb'
    with open_fn(file_path, mode) as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


def _load_img(file_path, is_gzip):
    """加载图像文件"""
    open_fn = gzip.open if is_gzip else open
    mode = 'rb'
    with open_fn(file_path, mode) as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, img_size)
    return data


def _convert_numpy(dataset_dir):
    """从指定目录加载并转换为 NumPy 数组"""
    dataset = {}
    
    for key, file_name in key_file.items():
        path, is_gzip = _get_file_path(dataset_dir, file_name)
        if path is None:
            raise FileNotFoundError(
                f"找不到 MNIST 文件: {file_name} 或 {file_name}.gz\n"
                f"请检查路径: {dataset_dir}"
            )
        print(f"Loading {file_name} ...")
        if 'label' in key:
            dataset[key] = _load_label(path, is_gzip)
        else:
            dataset[key] = _load_img(path, is_gzip)
        print("Done")
    
    return dataset


def _change_one_hot_label(X):
    """将标签转换为 one-hot 编码"""
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False, dataset_dir=None):
    """
    加载 MNIST 数据集

    Parameters
    ----------
    normalize : bool
        是否将像素值归一化到 0.0~1.0（默认 True）
    flatten : bool
        是否将图像展开为一维数组（默认 True，形状为 (N, 784)）
    one_hot_label : bool
        是否将标签转换为 one-hot 编码
    dataset_dir : str, optional
        数据集所在目录。若为 None，则使用默认路径。
        支持两种目录结构：
        1. 文件直接在目录下: train-images-idx3-ubyte 等
        2. PyTorch 风格，文件在 raw 子目录下: raw/train-images-idx3-ubyte 等
        支持 .gz 压缩和已解压的 .ubyte 文件

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if dataset_dir is None:
        dataset_dir = _default_dir
    
    dataset_dir = os.path.abspath(dataset_dir)
    
    # 检查是否有 pickle 缓存（仅当使用默认路径时）
    save_file = os.path.join(dataset_dir, 'mnist.pkl')
    if dataset_dir == _default_dir and os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = _convert_numpy(dataset_dir)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
