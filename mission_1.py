# 第二个任务：用神经网络预测点在圆内还是圆外——非线性神经网络MLP

# 本次任务调试最大的问题是我大小写混用
# 第一次跑，准确率稳定在0.5没变过，也就是说相当于随机猜测；ReLU死掉了
# 输入范围大约 [-1.5, 1.5]，He 初始化的 W1 元素服从 N(0, 1)（因为 fan_in=2, std=1），X @ W1 的典型值在 [-3, 3] 左右
# 偏置 b1=0.01 太微弱，无法将负值拉正，所以 z1 大量为负 → h1 = 0。
# h1全零 → dW2 = h1.T @ dz2 = 0 → W2不更新 → dz1 = dh1 * (z1>0) = 0 → W1, b1 也不更新。整个网络前端彻底卡死

# 第二次跑，train loss一直是1.4722，怀疑梯度消失了

# 任务实操要点：
# 隐藏层输出：z = ReLU(W1[x,y]^T + b1)，加入激活函数ReLU，使函数变弯
# 输出层logits: logits = W2 * z + b2
# 概率：p = softmax(logits)，从logits变成概率，输出0-1且自带比较

# 整体实现：
# 输入：点的坐标 (x, y) → 形状 (m, 2)
# 网络：2 → 4（ReLU） → 4（ReLU） → 2（Softmax）
# 输出：2 个类别的概率
# 损失：交叉熵（对 batch 取均值）
# 优化：梯度下降（手动实现反向传播）

# 1. 准备数据
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# 生成数据（圆形模式）
x,y = make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=42)
# 将噪声从0.2调成0.1，准确率从76%提升到96%

m = x.shape[0]
Y_onehot = np.zeros((m,2)) # 转换为独热编码，便于使用交叉熵
# 第一次做的时候直接tab了，变成了（m，10），报错
Y_onehot[range(m),y] = 1 # 整数标签变为二维矩阵，标识这个数据的真实类别



# 划分训练集和测试集 (8:2)
X_train, X_test, Y_train, Y_test = train_test_split(
    x, Y_onehot, test_size=0.2, random_state=42
)
y_test = np.argmax(Y_test, axis=1) # 转换回整数标签，方便计算准确率
print(f"训练样本数: {X_train.shape[0]}, 测试样本数: {X_test.shape[0]}")

# 数据标准化
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
X_train = (X_train - x_mean) / (x_std + 1e-8)
X_test  = (X_test  - x_mean) / (x_std + 1e-8)

# 2. 网络参数初始化
def he_init(fan_in, fan_out):
    """He 初始化：权重 ~ N(0, sqrt(2/fan_in))"""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

np.random.seed(0)  # 固定种子，保证每次运行结果一致
W1 = he_init(2, 4)
b1 = np.ones((1, 4)) * 0.5
W2 = he_init(4, 4)
b2 = np.ones((1, 4)) * 0.5
W3 = he_init(4, 2)
b3 = np.zeros((1, 2))


# 3. 设定超参数
lr = 0.01 # 学习率从0.1改到0.01，相应地epoch1000变1500，学习率小就多跑一点
epochs = 1500
batch_size = 128
n_train = X_train.shape[0]

train_losses = []
test_accs = []


# 4.训练
for epoch in range(epochs):
    # 每轮打乱训练数据
    perm = np.random.permutation(n_train)
    X_shuffled = X_train[perm]
    Y_shuffled = Y_train[perm]

    # mini_batch 梯度下降
    for i in range(0, n_train, batch_size):
        x_batch = X_train[i:i + batch_size]
        y_batch = Y_train[i:i + batch_size]

        # 前向传播
        z1 = x_batch @ W1 + b1  # (batch, 4)
        h1 = np.maximum(0, z1)  # ReLU
        z2 = h1 @ W2 + b2  # (batch, 4)
        h2 = np.maximum(0, z2)  # ReLU
        logits = h2 @ W3 + b3  # (batch, 2)

        # 数值稳定 softmax
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 交叉熵损失
        loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-8), axis=1))

        # 反向传播
        dlogits = probs - y_batch        # (batch, 2)  softmax+CE 的合体梯度
        dW3 = h2.T @ dlogits             # (4, 2)
        db3 = np.sum(dlogits, axis=0, keepdims=True)  # (1, 2)
        dh2 = dlogits @ W3.T             # (batch, 4)

        dz2 = dh2 * (z2 > 0)             # ReLU 反向：梯度流过时，z2<=0 的位置截断
        dW2 = h1.T @ dz2                 # (4, 4)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dh1 = dz2 @ W2.T

        dz1 = dh1 * (z1 > 0)
        dW1 = x_batch.T @ dz1            # (2, 4)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 参数更新（梯度下降）
        W3 -= lr * dW3
        b3 -= lr * db3
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1


    # 每个epoch后评估
    # 损失情况：
    z1_full = X_train @ W1 + b1
    h1_full = np.maximum(0, z1_full)
    z2_full = h1_full @ W2 + b2
    h2_full = np.maximum(0, z2_full)

    logits_full = h2_full @ W3 + b3
    logits_stable = logits_full - np.max(logits_full, axis=1, keepdims=True)
    probs_full = np.exp(logits_stable) / np.sum(np.exp(logits_stable), axis=1, keepdims=True)
    train_loss = -np.mean(np.sum(Y_train * np.log(probs_full + 1e-8), axis=1))
    train_losses.append(train_loss)

    # 测试准确率
    z1_test = X_test @ W1 + b1
    h1_test = np.maximum(0, z1_test)
    z2_test = h1_test @ W2 + b2
    h2_test = np.maximum(0, z2_test)
    logits_test = h2_test @ W3 + b3
    probs_test = np.exp(logits_test - np.max(logits_test, axis=1, keepdims=True))
    probs_test = probs_test / np.sum(probs_test, axis=1, keepdims=True)
    pred_test = np.argmax(probs_test, axis=1)
    acc = np.mean(pred_test == y_test)
    test_accs.append(acc)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Test Acc: {acc:.4f}")

print(f"\n最终测试集准确率: {test_accs[-1]:.4f}")

# 5. 可视化
# 没找到教程里给的py文件，干脆手搓了一个，在AI指导下完成
def predict(X):
    z1 = X @ W1 + b1
    h1 = np.maximum(0, z1)
    z2 = h1 @ W2 + b2
    h2 = np.maximum(0, z2)
    logits = h2 @ W3 + b3
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# 5.2 绘制决策边界和训练数据
plt.figure(figsize=(12, 5))
# 子图1：决策边界
plt.subplot(1, 2, 1)
# 生成网格点
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
# 对每个网格点预测类别
Z = predict(grid_points).reshape(xx.shape)

# 绘制填充的决策区域（半透明）
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
# 绘制原始数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(Y_train, axis=1),
            cmap='RdBu', edgecolor='k', s=30, label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
            cmap='RdBu', edgecolor='k', s=50, marker='s', label='Test')
plt.title("Decision Boundary (Circle Pattern)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

# 子图2：训练曲线
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_accs, label='Test Accuracy')
plt.title("Training Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()