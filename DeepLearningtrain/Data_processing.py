import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple
from tqdm import tqdm

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

def resample_action(action_data, target_length=100):
    """重采样单个动作片段到指定帧数"""
    indices = np.linspace(0, len(action_data)-1, target_length).astype(int)
    return action_data[indices]

def process_single_file(file_path: Path) -> pd.DataFrame:
    """处理单个CSV文件"""
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        data = df.values
        
        # 计算速度
        velocities = np.sqrt(np.sum(np.diff(data, axis=0)**2, axis=1))
        
        # 寻找分割点
        _, action_segments = find_start_and_action_segments(velocities)
        
        # 确保正好有5个动作片段
        if len(action_segments) != 5:
            print(f"警告: {file_path.name} 中找到 {len(action_segments)} 个动作片段，期望值为5")
            return None
        
        # 提取并重采样每个动作片段
        resampled_segments = []
        for start, end in action_segments:
            segment_data = data[start:end]
            resampled_segment = resample_action(segment_data, 100)
            resampled_segments.append(resampled_segment)
        
        # 将所有重采样后的动作合并成一个数组
        all_actions = np.vstack(resampled_segments)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(all_actions, columns=df.columns)
        
        return result_df
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def process_all_files(input_dir: str, output_dir: str):
    """处理所有CSV文件，保持目录结构"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 获取所有CSV文件及其相对路径
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)
    
    if not csv_files:
        print(f"在 {input_dir} 及其子目录中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个文件
    for file_path in tqdm(csv_files, desc="处理文件"):
        try:
            # 计算相对路径
            rel_path = file_path.relative_to(input_dir)
            # 构建输出路径
            output_path = output_dir / rel_path
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 处理文件
            result_df = process_single_file(file_path)
            
            if result_df is not None:
                # 检查数据形状
                print(f"\n文件 {rel_path} 的统计信息：")
                print(f"总行数: {len(result_df)} (预期500行)")
                print(f"特征数: {result_df.shape[1]} 列")
                
                # 保存处理后的文件
                result_df.to_csv(output_path, index=False)
                print(f"成功处理: {rel_path}")
            
        except Exception as e:
            print(f"处理 {file_path} 时出错: {str(e)}")
            continue
    
    print(f"\n处理完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    # 设置输入输出目录
    input_directory = "Yuren"    # 根目录
    output_directory = "Yuren_action"  # 输出根目录
    
    # 开始批量处理
    process_all_files(input_directory, output_directory)