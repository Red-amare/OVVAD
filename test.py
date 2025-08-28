import numpy as np
import cv2

# 路径改成你真实的某个视频和对应的特征文件
video_path = "/data/UCF_Crimes/Videos/Abuse/Abuse028_x264.mp4"
feat_path = "/data/UCF_Crimes/E_Features/Video/Abuse/Abuse028_x264__1.npy"

# 1. 读特征
feat = np.load(feat_path)  # shape [T, C]
T = feat.shape[0]
print("特征时间维度长度 T =", T)

# 2. 读视频帧数
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print("视频原始帧数 =", total_frames)

# 3. 计算 stride
stride = total_frames / T
print(f"每个特征覆盖的原始帧数 ≈ {stride:.2f}")
