import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r'file_path'
df = pd.read_csv(file_path, header=None)

# 提取横坐标（行数，从第二行开始）
x = df.index[1:]  # 从第二行开始的行数作为横坐标

# 提取纵坐标数据（每列的数据，从第二行开始）
data = df.iloc[1:, 1:]  # 从第二列开始的数据作为纵坐标

# 绘制折线图
plt.figure(figsize=(15, 8))

# 遍历每一列，绘制折线图
for col in data.columns:
    y = data[col]
    plt.plot(x, y, label=f'Column {col + 1}')  # +1是为了从1开始标记列

plt.title('折线图')
plt.xlabel('行数')
plt.ylabel('数值')
plt.ylim(df.iloc[1:, 1:].min().min(), df.iloc[1:, 1:].max().max())  # 设置纵坐标范围为数据的最小和最大值
plt.grid(True)

# 将图例放到图表的下方
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)  # ncol设置图例列数

# 保存图表为文件
plt.savefig('line_plot_with_legend_below.png', bbox_inches='tight')

# 显示图表
plt.show()
