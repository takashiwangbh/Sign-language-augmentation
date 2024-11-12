import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(r'Li\Best\middle.csv')
data = df.values

# 计算所有关键点的速度（相邻帧之间的欧氏距离）
velocities = np.sqrt(np.sum(np.diff(data, axis=0)**2, axis=1))

# 绘制速度曲线
plt.figure(figsize=(15, 5))
plt.plot(velocities)
plt.title('Li-Best-Middle')
plt.xlabel('帧数')
plt.ylabel('速度')
plt.grid(True)
plt.show()

# 打印一些基本信息
print(f"总帧数: {len(data)}")
print(f"数据形状: {data.shape}")