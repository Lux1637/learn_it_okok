import numpy as np


# 1.统一参数与初始化工具
class Parameter:
    """统一管理参数及其梯度"""

    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


def he_normal(fan_in, fan_out):
    """He 初始化 (适合 ReLU、SiLU、GELU)"""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


def xavier_normal(fan_in, fan_out):
    """Xavier 初始化 (适合 Sigmoid、Tanh、Softmax)"""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))


# 2.基础层与容器
class Sequential:
    """模块容器，串联各个网络层"""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'mode'): layer.mode = 'train'

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'mode'): layer.mode = 'test'


class Linear:
    """全连接层"""

    def __init__(self, in_features, out_features, init_fn=he_normal):
        self.W = Parameter(init_fn(in_features, out_features))
        self.b = Parameter(np.zeros((1, out_features)))
        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.W.data + self.b.data

    def backward(self, dout):
        # 梯度累加，支持各种灵活的batch操作
        self.W.grad += self.X.T @ dout
        self.b.grad += np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.data.T

    def get_params(self):
        return [self.W, self.b]


# 3.激活函数
class ReLU:
    def __init__(self): self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout):
        return dout * (self.X > 0)

    def get_params(self): return []


class SiLU:
    """SiLU激活函数"""

    def __init__(self): self.X = None; self.sigmoid = None

    def forward(self, X):
        self.X = X
        self.sigmoid = 1.0 / (1.0 + np.exp(-np.clip(X, -50, 50)))  # 防止溢出
        return X * self.sigmoid

    def backward(self, dout):
        silu = self.X * self.sigmoid
        return dout * (silu + self.sigmoid * (1.0 - silu))

    def get_params(self): return []


class GELU:
    """GELU激活函数"""

    def __init__(self): self.X = None; self.tanh = None

    def forward(self, X):
        self.X = X
        # Tanh 逼近公式
        self.tanh_arg = np.sqrt(2.0 / np.pi) * (X + 0.044715 * np.power(X, 3))
        self.tanh = np.tanh(self.tanh_arg)
        return 0.5 * X * (1.0 + self.tanh)

    def backward(self, dout):
        g_prime = np.sqrt(2.0 / np.pi) * (1.0 + 3 * 0.044715 * np.square(self.X))
        dtanh = 1.0 - np.square(self.tanh)
        dx = 0.5 * (1.0 + self.tanh) + 0.5 * self.X * dtanh * g_prime
        return dout * dx

    def get_params(self): return []


# 4.归一化与正则化 (LayerNorm & Dropout)
class LayerNorm:
    """层归一化 (Transformer 的核心基石)"""

    def __init__(self, dims, eps=1e-5):
        self.gamma = Parameter(np.ones((1, dims)))
        self.beta = Parameter(np.zeros((1, dims)))
        self.eps = eps
        self.X_hat, self.var = None, None

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        self.var = np.var(X, axis=-1, keepdims=True)
        self.X_hat = (X - mean) / np.sqrt(self.var + self.eps)
        return self.gamma.data * self.X_hat + self.beta.data

    def backward(self, dout):
        self.gamma.grad += np.sum(dout * self.X_hat, axis=0, keepdims=True)
        self.beta.grad += np.sum(dout, axis=0, keepdims=True)

        D = dout.shape[-1]
        dX_hat = dout * self.gamma.data
        # 矩阵形式的 LayerNorm 优雅反向传播
        dX = (D * dX_hat - np.sum(dX_hat, axis=-1, keepdims=True) -
              self.X_hat * np.sum(dX_hat * self.X_hat, axis=-1, keepdims=True)) / (D * np.sqrt(self.var + self.eps))
        return dX

    def get_params(self):
        return [self.gamma, self.beta]


class Dropout:
    """Dropout 正则化"""

    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.mode = 'train'

    def forward(self, X):
        if self.mode == 'train' and self.p > 0:
            self.mask = (np.random.rand(*X.shape) >= self.p) / (1.0 - self.p)
            return X * self.mask
        return X

    def backward(self, dout):
        if self.mode == 'train' and self.p > 0:
            return dout * self.mask
        return dout

    def get_params(self):
        return []



# 5.优化器
class AdamW:
    """AdamW 优化器"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0)

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None: continue

            # 1. 执行权重衰减 (L2 正则化)
            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data

            # 2. 更新一阶和二阶动量
            self.m[i] = b1 * self.m[i] + (1 - b1) * p.grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * (p.grad ** 2)

            # 3. 偏差修正
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)

            # 4. 更新参数
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# 6.损失函数
class CrossEntropyLoss:
    """数值稳定的交叉熵损失"""

    def __init__(self): self.probs, self.targets = None, None

    def __call__(self, logits, targets):
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.targets = targets
        return -np.mean(np.sum(targets * np.log(self.probs + 1e-8), axis=1))

    def backward(self):
        # 严格除以 batch_size，使标准梯度与 batch 大小解耦
        return (self.probs - self.targets) / self.targets.shape[0]


# 7.卷积核
def get_im2col_indices(x_shape, kh, kw, padding=1, stride=1):
    """
        生成高度优化的索引矩阵，这是 im2col 的灵魂。
    """
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    # 1：计算单个通道内卷积核像素的相对位置
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, C)
    j0 = np.tile(np.arange(kw), kh * C)

    # 2：计算输出特征图对应原图的绝对滑动偏移量
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    # 3：利用广播机制相加，得到全量索引
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)

def im2col(x, kh, kw, padding=1, stride=1):
        """
        将图像转换为二维矩阵
        返回维度: (C * kh * kw, N * out_h * out_w)
        """
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        k, i, j = get_im2col_indices(x.shape, kh, kw, padding, stride)

        # 核心高级索引：一步到位抠出所有窗口
        cols = x_padded[:, k, i, j]  # 形状: (N, C*kh*kw, out_h*out_w)
        C = x.shape[1]

        # 调整轴的顺序并展平
        cols = cols.transpose(1, 0, 2).reshape(C * kh * kw, -1)
        return cols

def col2im(cols, x_shape, kh, kw, padding=1, stride=1):
        """
        im2col 的完美逆运算：将二维梯度矩阵还原回四维图像梯度
        """
        N, C, H, W = x_shape
        p = padding
        x_padded = np.zeros((N, C, H + 2 * p, W + 2 * p))
        k, i, j = get_im2col_indices(x_shape, kh, kw, padding, stride)

        # 恢复维度以便用 np.add.at 累加梯度
        cols_reshaped = cols.reshape(C * kh * kw, N, -1).transpose(1, 0, 2)

        # 使用 add.at 处理重叠区域的梯度累加（重叠区域梯度相加）
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

        if p > 0:
            return x_padded[:, :, p:-p, p:-p]
        return x_padded

class Conv2D:

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # He 初始化（针对 ReLU 优化的标准初始化）
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        # 权重维度: (输出通道 O, 输入通道 C, 核高 K, 核宽 K)
        self.W = Parameter(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        self.b = Parameter(np.zeros((out_channels, 1)))

        # 缓存中间变量，供反向传播使用
        self.X = None
        self.X_col = None

    def forward(self, X):
        # 缓存输入 X
        self.X = X
        N, C, H, W = X.shape
        O, _, K, _ = self.W.data.shape

        # 1. 计算输出的特征图尺寸
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1

        # 2. 调用核心引擎：把四维图像变成二维大矩阵
        # 维度变换：(N, C, H, W) -> (C*K*K, N*H_out*W_out)
        self.X_col = im2col(X, K, K, self.padding, self.stride)

        # 3. 展平权重矩阵
        # 维度变换：(O, C, K, K) -> (O, C*K*K)
        W_flat = self.W.data.reshape(O, -1)

        # 4. 高效矩阵乘法 + 偏置项广播
        # (O, C*K*K) @ (C*K*K, N*H_out*W_out) -> (O, N*H_out*W_out)
        out = W_flat @ self.X_col + self.b.data

        # 5. 恢复成四维标准图像张量 (N, O, H_out, W_out)
        # 先拉回五维，把各个轴理顺，再调换轴的顺序（Transpose）
        out = out.reshape(O, N, H_out, W_out)
        out = out.transpose(1, 0, 2, 3)

        return out

    def backward(self, dout):
        # dout 原始维度: (N, O, H_out, W_out)
        N, O, H_out, W_out = dout.shape
        _, C, K, _ = self.W.data.shape

        # 1. 为了契合矩阵乘法，把 dout 转换为二维矩阵
        # (N, O, H_out, W_out) -> (O, N, H_out, W_out) -> (O, N*H_out*W_out)
        dout_flat = dout.transpose(1, 0, 2, 3).reshape(O, -1)

        # 2. 计算权重的梯度 dW
        # (O, N*H_out*W_out) @ (N*H_out*W_out, C*K*K) -> (O, C*K*K)
        dW_flat = dout_flat @ self.X_col.T
        self.W.grad += dW_flat.reshape(self.W.data.shape)  # 累加并恢复成四维权重形状

        # 3. 计算偏置的梯度 db (沿着一整行所有的列求和)
        self.b.grad += np.sum(dout_flat, axis=1, keepdims=True)

        # 4. 计算流向下游输入的列梯度 dX_col
        # W_flat 的转置: (C*K*K, O) @ dout_flat: (O, N*H_out*W_out) -> (C*K*K, N*H_out*W_out)
        W_flat = self.W.data.reshape(O, -1)
        dX_col = W_flat.T @ dout_flat

        # 5. 调用核心逆引擎：把二维梯度矩阵还原为四维图像梯度形状 (N, C, H, W)
        dX = col2im(dX_col, self.X.shape, K, K, self.padding, self.stride)

        return dX

    def get_params(self):
        return [self.W, self.b]


# 8.池化和batchnorm

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2, padding=0):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.X_shape = None
        self.max_idx = None
        self.X_col = None

    def forward(self, X):
        self.X_shape = X.shape  # 记住输入的原始形状 (N, C, H, W)
        N, C, H, W = X.shape
        kh = kw = self.pool_size

        # 巧妙设计：将N和C合并，把多通道图视作单通道，复用 im2col
        X_reshaped = X.reshape(N * C, 1, H, W)
        self.X_col = im2col(X_reshaped, kh, kw, self.padding, self.stride)  # (kh*kw, N*C*out_h*out_w)

        # 在每个滑动窗口（即每一列）中寻找最大值的行索引
        self.max_idx = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_idx, np.arange(self.max_idx.size)]

        # 恢复成标准的四维输出
        out_h = (H + 2 * self.padding - kh) // self.stride + 1
        out_w = (W + 2 * self.padding - kw) // self.stride + 1
        return out.reshape(N, C, out_h, out_w)

    def backward(self, dout):
        N, C, H, W = self.X_shape
        kh = kw = self.pool_size

        # 1：产生一个和前向一模一样的全零矩阵
        dX_col = np.zeros_like(self.X_col)

        # 2：独木桥原理：只有当时 forward 胜出的最大值位置，才能分到上游传下来的梯度
        dX_col[self.max_idx, np.arange(self.max_idx.size)] = dout.ravel()

        # 3：通过 col2im 把矩阵梯度平铺回图像形状
        dX = col2im(dX_col, (N * C, 1, H, W), kh, kw, self.padding, self.stride)
        return dX.reshape(self.X_shape)


class BatchNorm2D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.mode = 'train'
        self.gamma = Parameter(np.ones((1, num_features, 1, 1)))
        self.beta = Parameter(np.zeros((1, num_features, 1, 1)))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        # 前向传播时必须存留的中间状态
        self.X, self.X_hat, self.mean, self.var = None, None, None, None

    def forward(self, X):
        self.X = X  # 反向传播时需要用到原始X
        if self.mode == 'train':
            self.mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            self.var = np.var(X, axis=(0, 2, 3), keepdims=True)
            self.X_hat = (X - self.mean) / np.sqrt(self.var + self.eps)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var
        else:
            self.X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma.data * self.X_hat + self.beta.data

    def backward(self, dout):
        """
        顺着前向计算图，一步一步往回推导
        """
        # 1：算参数本身的梯度
        self.gamma.grad += np.sum(dout * self.X_hat, axis=(0, 2, 3), keepdims=True)
        self.beta.grad += np.sum(dout, axis=(0, 2, 3), keepdims=True)

        N, C, H, W = dout.shape
        M = N * H * W  # 归一化集合中包含的元素总数

        # 2：核心：全矩阵链式法则分解
        # 2.1: 传过加法和乘法层，得到 dx_hat
        dx_hat = dout * self.gamma.data

        # 2.2: 得到分母标准差的逆 inv_std
        inv_std = 1.0 / np.sqrt(self.var + self.eps)

        # 2.3: 算流向 inv_std 的梯度的分支
        x_mu = self.X - self.mean
        d_inv_std = np.sum(dx_hat * x_mu, axis=(0, 2, 3), keepdims=True)

        # 2.4: 算流向标准差std的梯度
        d_std = d_inv_std * (-inv_std ** 2)

        # 2.5: 算流向方差var的梯度
        d_var = d_std * 0.5 * inv_std

        # 2.6: 算流向平方差 (X-mu)^2 的梯度
        d_sq = np.ones_like(self.X) * (d_var / M)

        # 2.7: 汇总两条流向 (X-mu) 的梯度路径
        dx_mu1 = dx_hat * inv_std
        dx_mu2 = d_sq * 2.0 * x_mu
        dx_mu = dx_mu1 + dx_mu2

        # 2.8: 算流向均值mean的梯度并最终汇合输出流向X的总梯度dX
        d_mean = np.sum(dx_mu * (-1.0), axis=(0, 2, 3), keepdims=True)
        dX = dx_mu * 1.0 + np.ones_like(self.X) * (d_mean / M)

        return dX

    def get_params(self):
        return [self.gamma, self.beta]


class GlobalAvgPool2D:
    def __init__(self):
        self.in_shape = None

    def forward(self, X):
        self.in_shape = X.shape # 记住输入的维度 (N, C, H, W)
        # 把H和W维度的像素全部揉碎取平均值
        return np.mean(X, axis=(2, 3)) # 输出维度: (N, C)

    def backward(self, dout):
        N, C, H, W = self.in_shape
        # 反向传播：把接收到的(N, C)梯度平均平摊给当时H * W的每一个像素点
        return dout[:, :, None, None] * np.ones((N, C, H, W)) / (H * W)

    def get_params(self):
        return []


# 9.残差基本块
class BasicBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        # 第一层卷积：负责可能的降采样（stride=2）
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()

        # 第二层卷积：保持尺寸和通道不变
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()

        # 检查捷径（Shortcut）是否需要进行通道拉伸或尺寸缩放
        self.has_shortcut = (stride != 1 or in_channels != out_channels)
        if self.has_shortcut:
            # 用1x1卷积和BN来把输入的形状变成和输出一模一样
            self.shortcut_conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm2D(out_channels)

        self._mode = 'train'

    def forward(self, X):
        # 1：主通路前向
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        # 2：捷径通路前向
        if self.has_shortcut:
            identity = self.shortcut_conv.forward(X)
            identity = self.shortcut_bn.forward(identity)
        else:
            identity = X

        # 3：核心：残差相加，然后过最后的 ReLU
        self.out_add = out + identity
        return self.relu2.forward(self.out_add)

    def backward(self, dout):
        # 1：倒流穿过最后的 ReLU
        dout_add = self.relu2.backward(dout)

        # 2：梯度在加法处一分为二，各自倒流
        # 主通路倒流
        dout_main = self.bn2.backward(dout_add)
        dout_main = self.conv2.backward(dout_main)
        dout_main = self.relu1.backward(dout_main)
        dout_main = self.bn1.backward(dout_main)
        dX_main = self.conv1.backward(dout_main)

        # 捷径通路倒流
        if self.has_shortcut:
            dout_short = self.shortcut_bn.backward(dout_add)
            dX_short = self.shortcut_conv.backward(dout_short)
        else:
            dX_short = dout_add

        # 3：汇总两路返回给输入源头的总梯度
        return dX_main + dX_short

    def get_params(self):
        params = []
        params.extend(self.conv1.get_params())
        params.extend(self.bn1.get_params())
        params.extend(self.conv2.get_params())
        params.extend(self.bn2.get_params())
        if self.has_shortcut:
            params.extend(self.shortcut_conv.get_params())
            params.extend(self.shortcut_bn.get_params())
        return params

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        """当下游调用 block.mode = 'test' 时，内部的所有 BN 会自动同步切换"""
        self._mode = value
        self.bn1.mode = value
        self.bn2.mode = value
        if self.has_shortcut:
            self.shortcut_bn.mode = value
