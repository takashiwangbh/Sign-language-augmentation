import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split

# 数据集类定义
class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 模型定义
class GestureCNN3D(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN3D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.flatten_size = 64 * 100 * 21 * 3
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
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
    
    total_rows = 0
    time_steps =100
    num_nodes = 21
    num_features = 3
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data = df.values
        
        total_rows += data.shape[0]
        
        num_samples = data.shape[0] // time_steps
        if data.shape[0] % time_steps != 0:
            data = data[:num_samples * time_steps]
        
        data_reshaped = data.reshape(num_samples, time_steps, num_nodes, num_features)
        data_reshaped = np.expand_dims(data_reshaped, axis=1)
        
        data_list.append(data_reshaped)
        labels.extend([label] * num_samples)
    
    print(f"{folder_path} 文件夹读取的总行数: {total_rows}")
    return np.vstack(data_list), np.array(labels)

# 训练函数
def train_model(model, train_loader, test_loader, device, num_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    train_acc_list = []
    test_acc_list = []
    
    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        # 训练阶段
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            # 计算训练准确度
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # 计算当前epoch的训练准确度
        train_acc = train_correct / train_total
        train_acc_list.append(train_acc)
        
        # 评估阶段
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
        
        test_acc = test_correct / test_total
        test_acc_list.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # 保存准确率记录
    acc_df = pd.DataFrame({
        'Train Accuracy': train_acc_list,
        'Test Accuracy': test_acc_list
    })
    acc_df.to_csv('accuracy_history.csv', index=False)
    
    return train_acc_list, test_acc_list

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 读取数据
    parent_folders = ['Pu_action', 'Li_action', 'Yunhao_action', 'Wang_action']
    all_data = []
    all_labels = []
    
    for parent_folder in parent_folders:
        gesture_folders = [os.path.join(parent_folder, folder) 
                        for folder in os.listdir(parent_folder) 
                        if os.path.isdir(os.path.join(parent_folder, folder))]
        
        for label, folder_path in enumerate(gesture_folders):
            data, labels = load_data_from_folder(folder_path, label)
            all_data.append(data)
            all_labels.append(labels)
    
    # 合并数据
    all_data = np.vstack(all_data)
    all_labels = np.concatenate(all_labels)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = GestureDataset(X_train, y_train)
    test_dataset = GestureDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 创建模型
    num_classes = len(gesture_folders)
    model = GestureCNN3D(num_classes).to(device)
    
    # 训练模型
    train_acc_list, test_acc_list = train_model(model, train_loader, test_loader, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'gesture_model_pytorch_real_1000epoch_100step.pth')

if __name__ == '__main__':
    main()