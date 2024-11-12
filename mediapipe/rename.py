import os

# 指定文件夹路径
folder_path = r'E:\Signlunguagevideodata'  # 请替换为你的实际文件夹路径

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 构造完整的文件路径
    file_path = os.path.join(folder_path, filename)

    # 检查文件名是否是A.mp4、B.mp4或C.mp4，并进行相应的重命名
    if filename == 'A.mp4':
        new_filename = 'middle.mp4'
    elif filename == 'B.mp4':
        new_filename = 'right.mp4'
    elif filename == 'C.mp4':
        new_filename = 'left.mp4'
    else:
        continue  # 如果文件名不是A.mp4、B.mp4或C.mp4，则跳过

    # 构造新的文件路径
    new_file_path = os.path.join(folder_path, new_filename)
    
    # 检查新文件名是否已经存在，避免覆盖
    if os.path.exists(new_file_path):
        print(f"目标文件 {new_file_path} 已存在，跳过重命名 {filename}")
    else:
        # 重命名文件
        os.rename(file_path, new_file_path)
        print(f"文件重命名: {filename} -> {new_filename}")

print("文件重命名完成")
