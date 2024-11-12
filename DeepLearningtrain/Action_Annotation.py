import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_start_and_action_segments(velocities):
    window_size = 100
    static_threshold = 0.01  # 静止状态的阈值
    min_static_duration = 50  # 最小静止持续时间
    
    # 1. 先找到所有速度很小的区域
    is_static = velocities < static_threshold
    static_regions = []
    start = 0
    in_static = True  # 默认从静止开始
    
    # 2. 用滑动窗口来确认持续的静止状态
    for i in range(window_size, len(velocities)-window_size, window_size):
        window_mean = np.mean(velocities[i:i+window_size])
        
        if not in_static and window_mean < static_threshold:
            start = i
            in_static = True
        elif in_static and window_mean > static_threshold:
            if i - start >= min_static_duration:  # 确保静止持续足够长
                static_regions.append((start, i))
            in_static = False
    
    # 添加最后一个静止区域
    if in_static and len(velocities) - start >= min_static_duration:
        static_regions.append((start, len(velocities)))
    
    # 3. 保证至少有6个静止区域
    if len(static_regions) < 6:
        # 如果找不到足够的静止区域，均匀分配
        total_frames = len(velocities)
        segment_size = total_frames // 11  # 6个静止区域，5个动作区域
        static_regions = []
        for i in range(6):
            start = i * segment_size * 2
            end = start + segment_size
            static_regions.append((start, min(end, total_frames)))
    
    # 4. 如果有太多静止区域，选择最显著的6个
    if len(static_regions) > 6:
        # 计算每个静止区域的"静止程度"
        static_scores = []
        for start, end in static_regions:
            score = end - start  # 持续时间
            avg_velocity = np.mean(velocities[start:end])
            static_scores.append((start, end, score * (1/avg_velocity)))
        
        # 选择得分最高的6个
        static_scores.sort(key=lambda x: x[2], reverse=True)
        static_regions = [(start, end) for start, end, _ in static_scores[:6]]
        static_regions.sort()  # 按时间顺序排序
    
    # 5. 在静止区域之间标记动作区域
    action_segments = []
    for i in range(len(static_regions)-1):
        action_start = static_regions[i][1]
        action_end = static_regions[i+1][0]
        action_segments.append((action_start, action_end))
    
    return static_regions, action_segments

# 读取数据
df = pd.read_csv(r'D:\git\signlanguage\DeepLearningtrain\Wang\Best\middle.csv')
data = df.values
velocities = np.sqrt(np.sum(np.diff(data, axis=0)**2, axis=1))

# 找到所有区间
static_regions, action_segments = find_start_and_action_segments(velocities)

# 重采样动作片段
def resample_action(action_data, target_length=100):
    indices = np.linspace(0, len(action_data)-1, target_length).astype(int)
    return action_data[indices]

# 处理所有动作片段
resampled_segments = []
for start, end in action_segments:
    segment_data = data[start:end]
    resampled_segment = resample_action(segment_data, 100)
    resampled_segments.append(resampled_segment)

# 转换为numpy数组
resampled_segments = np.array(resampled_segments)

print("重采样后的数据形状:", resampled_segments.shape)

# 绘制结果
plt.figure(figsize=(10, 10))
plt.plot(velocities, linewidth=1.5)

# 标注开始手势区间（灰色）
for i, (start, end) in enumerate(static_regions):
    plt.axvspan(start, end, alpha=0.2, color='gray', label=f'开始手势{i+1}' if i==0 else "")

# 标注动作区间（彩色）
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i, (start, end) in enumerate(action_segments):
    plt.axvspan(start, end, alpha=0.3, color=colors[i], label=f'动作{i+1}')

plt.title('手势视频完整分割（6个开始手势 + 5个动作）', fontsize=16)
plt.xlabel('帧', fontsize=14)
plt.ylabel('速度', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(np.arange(0, len(velocities), 50), fontsize=12)
plt.yticks(fontsize=12)

plt.show()

# 打印区间信息
print("开始手势区间:")
for i, (start, end) in enumerate(static_regions):
    print(f"开始手势{i+1}: {start}-{end} (持续{end-start}帧)")

print("\n动作区间:")
for i, (start, end) in enumerate(action_segments):
    print(f"动作{i+1}: {start}-{end} (持续{end-start}帧)")