import numpy as np



# 1. 统一参数与初始化工具

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


# 2. 基础层与容器
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


# 4. 归一化与正则化 (LayerNorm & Dropout)
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



# 5. 优化器
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


# 6. 损失函数
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