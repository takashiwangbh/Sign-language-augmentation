import numpy as np
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
import tensorflow as tf

###################################读取文件

# 定义时间步长和节点数
time_steps = 10  # 根据需要调整
num_nodes = 21
num_features = 3

# 手势文件夹的父文件夹路径
parent_folder_path1 = 'Pu'
parent_folder_path2 = 'Li'
parent_folder_path3 = 'unity'
parent_folder_path4 = 'Yunhao'
parent_folder_path5 = 'Wang'


# 读取数据并��接
def load_data_from_folder(folder_path, label):
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    data_list = []
    labels = []


    total_rows = 0    #初始化行数计数器
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data = df.values
        
        total_rows += data.shape[0] #累加每个文件的行数
        
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

    print(f"{folder_path} 文件夹读取的总行数: {total_rows}")
    return np.vstack(data_list), np.array(labels)


# 获取所有手势的文件夹
gesture_folders1 = [os.path.join(parent_folder_path1, folder) for folder in os.listdir(parent_folder_path1) if os.path.isdir(os.path.join(parent_folder_path1, folder))]
gesture_folders2 = [os.path.join(parent_folder_path2, folder) for folder in os.listdir(parent_folder_path2) if os.path.isdir(os.path.join(parent_folder_path2, folder))]
gesture_folders3 = [os.path.join(parent_folder_path3, folder) for folder in os.listdir(parent_folder_path3) if os.path.isdir(os.path.join(parent_folder_path3, folder))]
gesture_folders4 = [os.path.join(parent_folder_path4, folder) for folder in os.listdir(parent_folder_path4) if os.path.isdir(os.path.join(parent_folder_path4, folder))]
gesture_folders5 = [os.path.join(parent_folder_path5, folder) for folder in os.listdir(parent_folder_path5) if os.path.isdir(os.path.join(parent_folder_path5, folder))]


all_data = []
all_labels = []

for label, folder_path in enumerate(gesture_folders1):
    data, labels = load_data_from_folder(folder_path, label)
    all_data.append(data)
    all_labels.append(labels)

for label, folder_path in enumerate(gesture_folders2):
    data, labels = load_data_from_folder(folder_path, label)
    all_data.append(data)
    all_labels.append(labels)


for label, folder_path in enumerate(gesture_folders3):
    data, labels = load_data_from_folder(folder_path, label)
    all_data.append(data)
    all_labels.append(labels)

for label, folder_path in enumerate(gesture_folders4):
    data, labels = load_data_from_folder(folder_path, label)
    all_data.append(data)
    all_labels.append(labels)


for label, folder_path in enumerate(gesture_folders5):
    data, labels = load_data_from_folder(folder_path, label)
    all_data.append(data)
    all_labels.append(labels)


# 合并所有数据
all_data = np.vstack(all_data)
all_labels = np.concatenate(all_labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



#####################模型训练#############################################
# 构建模型
# 创建 MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # 构建模型
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(time_steps, num_nodes, num_features, 1)),
        # MaxPooling3D((2, 2, 2)),
        Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        # MaxPooling3D((2, 2, 2)),
        Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        # MaxPooling3D((2, 2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(gesture_folders1), activation='softmax')  # 输出层，分类数等于手势数量
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


test_acc_list = []

# 训练模型
# 用for循环代替epochs

for i in range(100):
    model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test)
    test_acc_list.append(test_acc)

#保存test_acc_list到新的csv文件

test_acc_df = pd.DataFrame(test_acc_list, columns=['Test Accuracy'])
test_acc_df.to_csv('test_acc_list.csv_all', index=False)


# 输出模型的准确度
test_loss, test_acc = model.evaluate(X_test, y_test)



######   earlystopping #################################

###################################
#保存模型
model.save('test_gesture_Final_model_all.h5')