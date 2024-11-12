import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import os
from torch.utils.data import Dataset, DataLoader

# 数据集类
class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 修改后的模型类，与训练时保持一致
class GestureCNN3D(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN3D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout3d(0.2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout3d(0.2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout3d(0.2),
        )
        
        self.flatten_size = 64 * 100 * 21 * 3
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 数据加载函数
def load_data_from_folder(folder_path, label):
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    data_list = []
    labels = []
    
    time_steps = 100
    num_nodes = 21
    num_features = 3
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data = df.values
        
        num_samples = data.shape[0] // time_steps
        if data.shape[0] % time_steps != 0:
            data = data[:num_samples * time_steps]
        
        data_reshaped = data.reshape(num_samples, time_steps, num_nodes, num_features)
        data_reshaped = np.expand_dims(data_reshaped, axis=1)
        
        data_list.append(data_reshaped)
        labels.extend([label] * num_samples)
    
    return np.vstack(data_list), np.array(labels)

def predict_gestures(model_path, data_folder):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 获取训练时的文件夹顺序（确保标签一致性）
    train_folders = sorted(os.listdir('Pu'))
    folder_to_label = {folder: i for i, folder in enumerate(train_folders)}
    
    # 加载模型
    num_classes = len(train_folders)
    model = GestureCNN3D(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式，关闭Dropout
    
    # 加载测试数据
    all_data = []
    all_labels = []
    
    test_folders = sorted(os.listdir(data_folder))
    for folder in test_folders:
        if folder in folder_to_label:
            folder_path = os.path.join(data_folder, folder)
            if os.path.isdir(folder_path):
                data, _ = load_data_from_folder(folder_path, folder_to_label[folder])
                all_data.append(data)
                all_labels.extend([folder_to_label[folder]] * (len(data)))
    
    # 合并数据
    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)
    
    # 创建数据集和加载器
    test_dataset = GestureDataset(all_data, all_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 进行预测
    all_predictions = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = correct / total
    print(f"Predictions: {np.array(all_predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return all_predictions, accuracy

if __name__ == '__main__':
    model_path = 'gesture_model_pytorch_real_1000eopch_100step.pth'
    test_folder = 'Yuren_action'
    predict_gestures(model_path, test_folder)