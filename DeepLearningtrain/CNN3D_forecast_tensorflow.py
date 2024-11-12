import numpy as np
import pandas as pd
import glob
import os
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('test_gesture_Final_model_real')

# 进行预测的函数
def predict_gesture(model, data):
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

# 定义时间步长和节点数
time_steps = 10  # 根据需要调整
num_nodes = 21
num_features = 3

# 新数据的文件夹路径
parent_folder_path = 'Yuren'

def load_data_from_folder(folder_path, label):
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    data_list = []
    labels = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data = df.values
        
        # 确保数据总行数是时间步长的整数倍
        num_samples = data.shape[0] // time_steps
        if data.shape[0] % time_steps != 0:
            # 裁剪数据为10的整数倍
            data = data[:num_samples * time_steps]

            # raise ValueError(f"数据行数必须是时间步长的整数倍，文件：{file_path}")
        
        # 重塑数据
        data_reshaped = data.reshape(num_samples, time_steps, num_nodes, num_features)
        
        # 添加通道维度
        data_reshaped = np.expand_dims(data_reshaped, axis=-1)  # 形状变为 (样本数, 时间步长, 节点数, 3, 1)
        
        # 添加到数据列表
        data_list.append(data_reshaped)
        labels.extend([label] * num_samples)
    
    return np.vstack(data_list), np.array(labels)

# 获取所有手势的文件夹
gesture_folders = [os.path.join(parent_folder_path, folder) for folder in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, folder))]
all_data = []
all_labels = []

for label, folder_path in enumerate(gesture_folders):
    data, labels = load_data_from_folder(folder_path, label)
    all_data.append(data)
    all_labels.append(labels)

# 合并所有数据
all_data = np.vstack(all_data)
all_labels = np.concatenate(all_labels)


# 对新数据进行预测
predicted_labels = predict_gesture(model, all_data)

# 输出预测结果
print(predicted_labels)

#输出准确率
accuracy = np.mean(predicted_labels == all_labels)
print(f"Accuracy: {accuracy}")
