#本py文件为结合本章所有知识的训练代码汇总
import numpy as np
from collections import OrderedDict
# 加载数据集
# 加载本地MNIST数据集（PNG图片 + txt标签，带npy缓存）
import os
from PIL import Image

MNIST_DIR = './mnist'
CACHE_DIR = os.path.join(MNIST_DIR, 'cache')  # 缓存目录

def load_labels_from_txt(filepath):
    """从txt文件加载标签，格式：每行 '图片编号\t标签值'"""
    labels_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            idx, label = int(parts[0]), int(parts[1])
            labels_dict[idx] = label
    return labels_dict

def load_images_from_dir(img_dir, num_images):
    """从目录中按编号顺序加载PNG图片，展平为(N, 784)"""
    images = np.zeros((num_images, 784), dtype=np.uint8)
    for i in range(num_images):
        img_path = os.path.join(img_dir, f'{i}.png')
        img = Image.open(img_path).convert('L')  # 转灰度图
        images[i] = np.array(img).flatten()
    return images

def one_hot(labels, num_classes=10):
    """将标签转为one-hot编码"""
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels

def _load_raw_data():
    """加载原始数据（优先读缓存，没有则从PNG读取并生成缓存）"""
    cache_files = {
        'train_images': os.path.join(CACHE_DIR, 'train_images.npy'),
        'train_labels': os.path.join(CACHE_DIR, 'train_labels.npy'),
        'test_images': os.path.join(CACHE_DIR, 'test_images.npy'),
        'test_labels': os.path.join(CACHE_DIR, 'test_labels.npy'),
    }
    
    # 检查缓存是否全部存在
    if all(os.path.exists(f) for f in cache_files.values()):
        print('检测到npy缓存，直接加载...')
        train_images = np.load(cache_files['train_images'])
        train_labels = np.load(cache_files['train_labels'])
        test_images = np.load(cache_files['test_images'])
        test_labels = np.load(cache_files['test_labels'])
        print('缓存加载完成！')
        return train_images, train_labels, test_images, test_labels
    
    # 缓存不存在，从PNG和txt加载
    num_train = 55000
    num_test = 10000
    
    print('首次加载，正在读取PNG图片（之后会使用npy缓存）...')
    
    # 加载标签
    print('正在加载标签...')
    train_labels_dict = load_labels_from_txt(os.path.join(MNIST_DIR, 'train_labs.txt'))
    test_labels_dict = load_labels_from_txt(os.path.join(MNIST_DIR, 'test_labs.txt'))
    train_labels = np.array([train_labels_dict[i] for i in range(num_train)], dtype=np.uint8)
    test_labels = np.array([test_labels_dict[i] for i in range(num_test)], dtype=np.uint8)
    
    # 加载图片
    print(f'正在加载训练集图片({num_train}张)，请稍候...')
    train_images = load_images_from_dir(os.path.join(MNIST_DIR, 'train'), num_train)
    print(f'正在加载测试集图片({num_test}张)，请稍候...')
    test_images = load_images_from_dir(os.path.join(MNIST_DIR, 'test'), num_test)
    
    # 保存为npy缓存
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_files['train_images'], train_images)
    np.save(cache_files['train_labels'], train_labels)
    np.save(cache_files['test_images'], test_images)
    np.save(cache_files['test_labels'], test_labels)
    print(f'已将数据缓存到 {CACHE_DIR}/ 目录')
    
    return train_images, train_labels, test_images, test_labels

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    从本地 ./mnist 目录加载MNIST数据集（带npy缓存加速）
    
    首次运行：读取PNG图片 -> 存为npy缓存（约几十秒）
    之后运行：直接读npy缓存（约1秒）
    
    Parameters
    ----------
    normalize : bool
        是否将像素值归一化到 0.0~1.0
    flatten : bool
        是否将图像展平为一维数组(784,)，False则保持(1,28,28)
    one_hot_label : bool
        是否将标签转为one-hot编码
    
    Returns
    -------
    (train_images, train_labels), (test_images, test_labels)
    """
    train_images, train_labels, test_images, test_labels = _load_raw_data()
    
    # 归一化：像素值从 0~255 缩放到 0.0~1.0
    if normalize:
        train_images = train_images.astype(np.float64) / 255.0
        test_images = test_images.astype(np.float64) / 255.0
    
    # 不展平时reshape为 (N, 1, 28, 28)
    if not flatten:
        train_images = train_images.reshape(-1, 1, 28, 28)
        test_images = test_images.reshape(-1, 1, 28, 28)
    
    # one-hot编码
    if one_hot_label:
        train_labels = one_hot(train_labels)
        test_labels = one_hot(test_labels)
    
    return (train_images, train_labels), (test_images, test_labels)

# 加载数据集
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)
print(f'训练集: images={x_train.shape}, labels={y_train.shape}')
print(f'测试集: images={x_test.shape}, labels={y_test.shape}')
print(f'单张图像shape: {x_train[0].shape}, 像素值范围: [{x_train.min():.1f}, {x_train.max():.1f}]')


# 损失函数、激活函数和线性层
class SoftmaxWithLoss():
    def __init__(self,):
        self.x = None
        self.y_true = None
        self.loss = None

    @staticmethod
    def softmax(x:np.array):
        # 这里用XW的形式
        # softmax = exp(Xi-X_max)/sum(exp(Xj-X_max))
        return np.exp(x-np.max(x,axis = -1,keepdims=True))/np.sum(np.exp(x-np.max(x,axis = -1,keepdims=True)),axis=-1,keepdims=True)
    
    @staticmethod
    def cross_entropy_loss(y_pred:np.array,y_true:np.array):
        # 输出的每个维度都要做一次交叉熵的计算
        # 批次内所有的维度做完交叉熵求和取平均就是最后的交叉熵
        # entropy = -1/n*(sum(sum(y_true*log(y_pred))))
        if y_true.ndim == 1:
            y_true = y_true.reshape(1,y_true.size)
            y_pred = y_pred.reshape(1,y_pred.size)
        batch = y_true.shape[0]
        clip = 1e-7
        return -1/batch*np.sum(np.sum(y_true*np.log(y_pred+clip),axis = -1))

    def forward(self,x,y_true):
        self.y_true = y_true
        self.x = self.softmax(x)
        self.loss = self.cross_entropy_loss(y_true = y_true,y_pred = self.x)

        return self.loss

    def backward(self,dz=1):
        batch = self.y_true.shape[0]
        return (self.x-self.y_true)/batch

class LinearLayer():
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.w) + self.b
        return out

    def backward(self,dz):
        dx = np.dot(dz,self.w.T)
        self.dw = np.dot(self.x.T,dz)
        self.db = np.sum(dz,axis = 0)
        return dx

class SigmoidLayer():
    def __init__(self,):
        self.out = None

    def forward(self,x):
        out = 1/(np.exp(-1*x)+1)
        self.out = out
        return out

    def backward(self,dz):
        # f'(x) = f(x)(1-f(x))
        dx = dz*self.out*(1-self.out)
        return dx

# 模型网络
class Model():
    def __init__(self,input_dim,hidden_dim,output_dim=10):
        # weight
        self.net_params = {}
        self.net_params['w1'] = np.random.randn(input_dim,hidden_dim)
        self.net_params['b1'] = np.zeros(hidden_dim)
        self.net_params['w2'] = np.random.randn(hidden_dim,output_dim)
        self.net_params['b2'] = np.zeros(output_dim)

        # layers
        self.layers = OrderedDict() # 一定要有序才能反向传播（当然这里只有单路径）
        self.layers['linear1'] = LinearLayer(self.net_params['w1'],self.net_params['b1'])
        self.layers['sigmoid'] = SigmoidLayer()
        self.layers['linear2'] = LinearLayer(self.net_params['w2'],self.net_params['b2'])

        self.cross_entropy = SoftmaxWithLoss()

    def forward(self,x):
        # 前向传播
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def accuracy(self,x,y):
        x = self.forward(x)
        pred = np.argmax(x,axis=-1)
        if y.ndim != 1:
            y = np.argmax(y,axis=-1)
        
        acc = np.sum(y==pred) / float(x.shape[0])
        return acc

    def loss(self,x,y):
        x = self.forward(x)
        loss = self.cross_entropy.forward(x,y)
        # 仅用于记录
        return loss

    def gradient(self,x,y):
        self.loss(x,y)
        dL = 1
        dout = self.cross_entropy.backward(dL)

        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grad = {}
        grad['w1'] = self.layers['linear1'].dw
        grad['b1'] = self.layers['linear1'].db
        grad['w2'] = self.layers['linear2'].dw
        grad['b2'] = self.layers['linear2'].db

        return grad




from tqdm import tqdm

# ======================== 超参数 ========================
epoch = 1000                # PyTorch: num_epochs = 1000
train_size = x_train.shape[0]
batch_size = 100            # PyTorch: DataLoader(dataset, batch_size=100, shuffle=True)
lr =0.8                     # PyTorch: optimizer = optim.SGD(model.parameters(), lr=0.8)

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

# ======================== 模型初始化 ========================
# PyTorch等价:
#   model = nn.Sequential(
#       nn.Linear(784, 30),     ← linear1 + b1
#       nn.Sigmoid(),           ← sigmoid
#       nn.Linear(30, 10),      ← linear2 + b2
#   )
#   criterion = nn.CrossEntropyLoss()   ← SoftmaxWithLoss（PyTorch的CrossEntropyLoss自带softmax）
#   optimizer = optim.SGD(model.parameters(), lr=0.8)
network = Model(input_dim=784,hidden_dim=30,output_dim=10)

pbar = tqdm(range(epoch), desc='Training')
for i in pbar:
    # ============ 1. 采样 mini-batch ============
    # choice实现花式索引Fancy Indexing，按数组获取对应数据
    # PyTorch: for train_data, label in DataLoader(dataset, batch_size=100, shuffle=True):
    #   DataLoader自动shuffle+分batch，这里手动random.choice等价于有放回采样
    train_index = np.random.choice(train_size,batch_size)
    train_data = x_train[train_index]
    label = y_train[train_index]

    # ============ 2. 前向传播 + 反向传播 ============
    # PyTorch等价（拆成3步）:
    #   output = model(train_data)           ← 前向传播
    #   loss = criterion(output, label)      ← 计算损失
    #   loss.backward()                      ← 反向传播，自动计算所有参数的梯度
    grad = network.gradient(train_data,label)

    # print("w1 grad max:", np.max(np.abs(grad['w1'])))
    # print("w2 grad max:", np.max(np.abs(grad['w2'])))

    # ============ 3. 参数更新 ============
    # PyTorch等价:
    #   optimizer.step()       ← 用梯度更新所有参数（内部做 param -= lr * param.grad）
    #   optimizer.zero_grad()  ← 清零梯度（PyTorch梯度会累加，必须手动清零）
    # 注意：我们这里不需要zero_grad，因为backward里dw/db是赋值(=)而非累加(+=)，每次自动覆盖
    for key in ('w1','b1','w2','b2'):
        network.net_params[key] -= lr*grad[key]
    
    # ============ 4. 记录 loss ============
    # PyTorch: 通常在前向传播时就拿到了loss.item()，不需要再算一次
    #   train_loss = loss.item()
    #   with torch.no_grad():
    #       test_loss = criterion(model(x_test), y_test).item()
    train_loss = network.loss(train_data,label)
    test_loss = network.loss(x_test,y_test)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    # 更新进度条显示当前loss
    pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'test_loss': f'{test_loss:.4f}'})

    # ============ 5. 记录 accuracy ============
    # PyTorch等价:
    #   with torch.no_grad():             ← 推理时关闭梯度计算，省显存
    #       pred = model(x_test).argmax(dim=-1)
    #       acc = (pred == y_test).float().mean().item()
    if i % 10 == 0:
        train_acc_list.append(network.accuracy(train_data,label))
        test_acc_list.append(network.accuracy(x_test,y_test))

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：Loss 曲线
ax1.plot(range(len(train_loss_list)), train_loss_list, label='train loss')
ax1.plot(range(len(test_loss_list)), test_loss_list, label='test loss')
ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True)

# 右图：Accuracy 曲线（每10个iteration记录一次）
x_acc = list(range(0, len(train_acc_list) * 10, 10))
ax2.plot(x_acc, train_acc_list, marker='o', markersize=3, label='train acc')
ax2.plot(x_acc, test_acc_list, marker='s', markersize=3, label='test acc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_title('Accuracy Curve')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
