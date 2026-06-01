import numpy as np
import time
import os
import pickle

# 1.导入做好的底层零件
from my_dl_lib import AdamW, CrossEntropyLoss, Linear, ReLU

try:
    from my_dl_lib import Conv2D, BatchNorm2D, BasicBlock, GlobalAvgPool2D
except ImportError:

    print("[警告] 请确保 Conv2D, BatchNorm2D, BasicBlock, GlobalAvgPool2D 已在 my_dl_lib 中实现。")


# 2.组装
class DummyResNet:
    """
    轻量级 ResNet 架构
    输入维度: (N, 3, 32, 32) -> 输出维度: (N, 100)
    """

    def __init__(self, num_classes=100):
        # 根节点输入层: (N, 3, 32, 32) -> (N, 16, 32, 32)
        self.conv1 = Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2D(num_features=16)
        self.relu1 = ReLU()

        # 1:保持尺寸不变 -> (N, 16, 32, 32)
        self.stage1_1 = BasicBlock(in_channels=16, out_channels=16, stride=1)
        self.stage1_2 = BasicBlock(in_channels=16, out_channels=16, stride=1)

        # 2:空间尺寸减半，通道数加倍 -> (N, 32, 16, 16)
        self.stage2_1 = BasicBlock(in_channels=16, out_channels=32, stride=2)
        self.stage2_2 = BasicBlock(in_channels=32, out_channels=32, stride=1)

        # 3:空间尺寸再减半，通道数再加倍 -> (N, 64, 8, 8)
        self.stage3_1 = BasicBlock(in_channels=32, out_channels=64, stride=2)
        self.stage3_2 = BasicBlock(in_channels=64, out_channels=64, stride=1)

        # 全局平均池化与全连接分类器
        self.pool = GlobalAvgPool2D()  # -> (N, 64)
        self.fc = Linear(in_features=64, out_features=num_classes)

        # 将所有层存入列表，方便管理
        self.layers = [
            self.conv1, self.bn1, self.relu1,
            self.stage1_1, self.stage1_2,
            self.stage2_1, self.stage2_2,
            self.stage3_1, self.stage3_2,
            self.pool, self.fc
        ]

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
            if hasattr(layer, 'get_params'):
                params.extend(layer.get_params())
        return params

    def train(self):
        """切换为训练模式：激活 BatchNorm 的 running 统计量更新"""
        for layer in self.layers:
            if hasattr(layer, 'mode'):
                layer.mode = 'train'

    def eval(self):
        """切换为评估模式：固定 BatchNorm 的 running 统计量"""
        for layer in self.layers:
            if hasattr(layer, 'mode'):
                layer.mode = 'test'


# 3.小范围过拟合数据
def load_overfit_data(num_samples=100):
    """
    第一步：生成固定的小样本数据。
    """
    np.random.seed(42)
    # 模拟 100 张 32x32 的 RGB 图像
    X_mini = np.random.randn(num_samples, 3, 32, 32)

    # 模拟 100 类的 One-hot 标签
    mock_labels = np.random.randint(0, 100, size=num_samples)
    Y_mini = np.zeros((num_samples, 100))
    Y_mini[range(num_samples), mock_labels] = 1

    print(f"=== 已成功加载过拟合专用小数据集 ===")
    print(f"数据维度: {X_mini.shape} | 标签维度: {Y_mini.shape}")
    return X_mini, Y_mini



# 4.主训练逻辑
def main():
    # 超参数设置（针对过拟合高度优化）
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001  # 自制网络初试 AdamW，先用温和的学习率
    NUM_SAMPLES = 100  # 过拟合样本数

    # 加载数据
    X_train, Y_train = load_overfit_data(num_samples=NUM_SAMPLES)
    y_true_labels = np.argmax(Y_train, axis=1)

    # 实例化模型、优化器和损失函数
    model = DummyResNet(num_classes=100)
    optimizer = AdamW(model.get_params(), lr=LEARNING_RATE, weight_decay=0.0)  # 过拟合时关掉权重衰减
    criterion = CrossEntropyLoss()

    print("\n开始进行小样本过拟合测试...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()  # 确保进入训练模式

        # 打乱数据
        perm = np.random.permutation(NUM_SAMPLES)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch 循环
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            x_batch = X_shuffled[i:i + BATCH_SIZE]
            y_batch = Y_shuffled[i:i + BATCH_SIZE]

            # 1：前向传播
            logits = model.forward(x_batch)
            loss = criterion(logits, y_batch)
            epoch_loss += loss
            num_batches += 1

            # 2：反向传播
            optimizer.zero_grad()
            dout = criterion.backward()
            model.backward(dout)

            # 3：更新权重
            optimizer.step()

        # 计算当前 Epoch 的平均损失
        avg_loss = epoch_loss / num_batches

        # 每 10 个 Epoch 评估一次在训练集上的准确率
        if epoch % 10 == 0 or epoch == 1:
            model.eval()  # 评估时切换模式
            train_logits = model.forward(X_train)
            pred_labels = np.argmax(train_logits, axis=1)
            train_acc = np.mean(pred_labels == y_true_labels)

            print(f"Epoch [{epoch:3d}/{EPOCHS}] | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")

            # 终点检查：如果已经完美过拟合，提早结束
            if train_acc >= 0.99 and avg_loss < 0.05:
                print(f"\n🎉 恭喜！模型在第 {epoch} 轮达成完美过拟合！说明底层算子与反向传播完全正常。")
                break


    # 5.保存 Checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_weights': [p.data for p in model.get_params()],
        'optimizer_m': optimizer.m,
        'optimizer_v': optimizer.v
    }

    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/resnet_overfit.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"\n进度已保存至 checkpoints/resnet_overfit.pkl (耗时: {time.time() - start_time:.2f}秒)")


if __name__ == '__main__':
    main()