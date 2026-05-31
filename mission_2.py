# 第三个任务：完善我的小型学习库

# 在建立自己的库之后，重写一遍mission1的拟合过程
# 依旧调试x大小写。我真是服了。。。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

#导入我的小型学习库
from my_dl_lib import Sequential, Linear, ReLU, GELU, SiLU, LayerNorm, Dropout, AdamW, CrossEntropyLoss

# 1.准备数据
x, y = make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=42)
m = x.shape[0]
Y_onehot = np.zeros((m, 2))
Y_onehot[range(m), y] = 1

X_train, X_test, Y_train, Y_test = train_test_split(x, Y_onehot, test_size=0.2, random_state=42)
y_test = np.argmax(Y_test, axis=1)

# 数据标准化
x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - x_mean) / (x_std + 1e-8)
X_test = (X_test - x_mean) / (x_std + 1e-8)


# 2. 搭建网络
model = Sequential([
    Linear(2, 4),
    LayerNorm(4),  # 加入归一化
    GELU(),  # 换成GELU
    Linear(4, 4),
    LayerNorm(4),
    GELU(),
    Dropout(0.1),  # 加个正则化dropout
    Linear(4, 2)
])

# 3.初始化解耦的优化器和损失函数
optimizer = AdamW(model.get_params(), lr=0.02, weight_decay=0.01)  #整合AdamW和L2
criterion = CrossEntropyLoss()

# 4.训练循环
epochs = 400  #Adam收敛极快，不需要跑那么多轮
batch_size = 128
train_losses, test_accs = [], []

for epoch in range(epochs):
    model.train()  #开启 Train 模式 (影响 Dropout)
    perm = np.random.permutation(X_train.shape[0])
    X_shuffled, Y_shuffled = X_train[perm], Y_train[perm]

    for i in range(0, X_train.shape[0], batch_size):
        x_batch = X_shuffled[i:i + batch_size]
        y_batch = Y_shuffled[i:i + batch_size]

        # 前向
        logits = model.forward(x_batch)
        loss = criterion(logits, y_batch)

        # 反向
        optimizer.zero_grad()
        dout = criterion.backward()
        model.backward(dout)

        # 更新
        optimizer.step()

    # 评估阶段
    model.eval()  # 关闭开销或随机性 (影响 Dropout)
    train_loss = criterion(model.forward(X_train), Y_train)
    test_logits = model.forward(X_test)
    test_acc = np.mean(np.argmax(test_logits, axis=1) == y_test)

    train_losses.append(train_loss)
    test_accs.append(test_acc)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

print(f"\n最终测试集准确率: {test_accs[-1]:.4f}")


# 5.手搓的可视化部分
def predict(X):
    model.eval()
    logits = model.forward(X)
    return np.argmax(logits, axis=1)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(Y_train, axis=1), cmap='RdBu', edgecolor='k', s=30, label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdBu', edgecolor='k', s=50, marker='s', label='Test')
plt.title("Decision Boundary (With our My_DL_Lib)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_accs, label='Test Accuracy')
plt.title("Training Curves (AdamW Optimizer)")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()