import matplotlib.pyplot as plt
import numpy as np

# 创建一些数据
x = np.linspace(-5, 5, 100)
y = np.sin(x) / x

# 绘制图形
plt.plot(x, y)

# 在y轴的正负无穷大处添加标记
plt.axhline(y=np.inf, color='r', linestyle='--')
plt.axhline(y=-np.inf, color='r', linestyle='--')

# 显示图形
plt.show()
