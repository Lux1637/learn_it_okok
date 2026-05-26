# 第一个任务：拟合一条直线
# y = a * x + b, a:weight, b: bias
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# 1.准备数据
data = load_diabetes()
x = data.data[:, 2] # BMI 特征，已标准化
y_raw = data.target.reshape(-1, 1) # 目标值，未标准化，转为列向量
# 为了使用真实的数据进行训练，使用了sklearn自带的数据库
# 第一次做的时候由于x进行了标准化，但相应的y没有，训练了1000个epoch之后，loss仍然没有下降的趋势

scaler_y = StandardScaler() # 对y进行标准化（均值为0，方差为1）
y_true = scaler_y.fit_transform(y_raw).flatten()  # 变回一维数组

# 2. 随机初始化和超参数
np.random.seed(42)    # 固定随机种子，方便复现
a = np.random.randn()
b = np.random.randn()
print(f"初始随机数据：a = {a:.4f}, b = {b:.4f}")

learning_rate = 0.01 # 设定两个超参数
epochs = 2000
loss_history = [] # 记录损失

# 3. 训练循环
for epoch in range(epochs):
    y_pre = a * x + b # 带入模型
    loss = np.mean((y_pre - y_true) ** 2) #MSE
    loss_history.append(loss)

    # 梯度下降
    grad_a = 2 * np.mean((y_pre - y_true) * x)
    grad_b = 2 * np.mean(y_pre - y_true)

    # 调整参数
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b

    # 每100轮打印一次看当前情况
    if (epoch + 1) % 100 == 0:
        print(f"epoch: {epoch+1:4d}, loss: {loss:.6f}, a: {a:.4f}, b: {b:.4f}")

# 4. 预测 x=5（注意：预测值也要反标准化）
x_new = 5
y_new_scaled = a * x_new + b               # 这是标准化后的预测值
y_new = scaler_y.inverse_transform([[y_new_scaled]])[0, 0]   # 变回原始尺度
# 这个地方同样是在第一次尝试后查找资料改正的

print(f"训练后，x_new = 5 时，预测 y_new = {y_new:.4f}（真实疾病进展指标）")