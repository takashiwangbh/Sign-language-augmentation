import os
import cv2
import mediapipe as mp
import pandas as pd

# 打印当前工作目录
print("当前工作目录:", os.getcwd())

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 视频路径
video_path = r'video_path'  # 请替换为你的实际视频路径

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"无法打开视频文件: {video_path}")
    exit()

# 用于存储关键点的列表
keypoints_list = []
frame_count = 0  # 记录处理的帧数

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    print(f"正在处理第 {frame_count} 帧")

    # 将BGR图像转换为RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像并检测手部
    results = hands.process(image)

    if results.multi_hand_landmarks:
        print(f"第 {frame_count} 帧检测到手部关键点")
        for hand_landmarks in results.multi_hand_landmarks:
            frame_keypoints = []
            for landmark in hand_landmarks.landmark:
                frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
            keypoints_list.append(frame_keypoints)
    else:
        print(f"第 {frame_count} 帧未检测到手部关键点")

cap.release()

# 检查是否检测到任何关键点
if keypoints_list:
    # 将关键点列表转换为DataFrame
    df = pd.DataFrame(keypoints_list)

    # 获取视频文件所在文件夹
    video_folder = os.path.dirname(video_path)

    # 从视频路径中提取文件名（不带扩展名）
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # 构造CSV文件的保存路径
    output_path = os.path.join(video_folder, f'{video_filename}.csv')

    # 将DataFrame保存为CSV文件
    df.to_csv(output_path, index=False)
    print(f"CSV文件已保存到 {output_path}")
else:
    print("未检测到任何手部关键点，无法生成CSV文件。")
