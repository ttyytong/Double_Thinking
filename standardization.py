import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

# 读取txt文档中的数据
file_path = '/ssd0/tyt/CogVLM/training_loss_values.txt'  # 替换为你的txt文档路径
with open(file_path, 'r') as file:
    lines = file.readlines()
    # 假设数据是用逗号分隔的
    losses = [float(line.strip().split(',')[1]) for line in lines]

# 将数据标准化到0-1之间
min_loss = min(losses)
max_loss = max(losses)
normalized_losses = [(loss - min_loss) / (max_loss - min_loss) for loss in losses]

# 生成迭代次数，假设迭代次数从0到1000
iterations = np.linspace(0, 1200, len(normalized_losses))

# 使用Savitzky-Golay滤波器对数据进行平滑处理
window_length = 51  # 窗口长度（必须是奇数）
polyorder = 3  # 多项式阶数
smoothed_losses = savgol_filter(normalized_losses, window_length, polyorder)

# 使用样条插值法对平滑后的数据进行进一步平滑处理
x_smooth = np.linspace(iterations.min(), iterations.max(), 300)
spl = make_interp_spline(iterations, smoothed_losses, k=3)
y_smooth = spl(x_smooth)

# 绘制平滑后的数据曲线
plt.figure(figsize=(10, 5))
plt.plot(x_smooth, y_smooth, label='Normalized Loss')
plt.xlabel('Iterations')
plt.ylabel('Normalized Loss')
plt.title('Training Loss Curve ')
plt.legend()
plt.savefig('training_loss_curve3.png')
plt.show()


