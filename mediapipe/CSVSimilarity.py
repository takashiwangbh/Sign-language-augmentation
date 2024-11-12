import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file1 = pd.read_csv(r'D:\git\signlanguage\DeepLearningtrain\Pu\Y\middle.csv').values
file2 = pd.read_csv(r'D:\git\signlanguage\DeepLearningtrain\Li\Y\middle.csv').values

# 确保两个文件具有相同的维度
min_rows = min(len(file1), len(file2))
min_cols = min(file1.shape[1], file2.shape[1])
file1 = file1[:min_rows, :min_cols]
file2 = file2[:min_rows, :min_cols]

# 计算每行每个元素的距离
distances = np.abs(file1 - file2)

# 绘制每一行的平均距离
average_distances = np.mean(distances, axis=1)  # 按行计算平均距离

# 绘制平均距离的折线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, min_rows + 1), average_distances, marker='o', color='b')  # 横坐标从1开始
plt.title('Average Element-wise Distance per Row', fontsize=14)
plt.xlabel('Row Number', fontsize=12)  # 横坐标为"Row Number"
plt.ylabel('Average Distance', fontsize=12)
plt.xticks(range(1, min_rows + 1))  # 设置横坐标标签
plt.show()
