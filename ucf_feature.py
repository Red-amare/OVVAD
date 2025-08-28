import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import open_clip
from PIL import Image

# ======================== 配置区 ===========================
DATA_ROOT = "/data/UCF_Crimes/Videos"    # 按你的目录结构修改
FEAT_SAVE_ROOT = "/home/cxa/ucf"      # 输出特征保存目录
FRAME_INTERVAL = 16

CLIP_MODEL = "ViT-B-16"
PRETRAINED = "/home/cxa/clip_models/open_clip_pytorch_model.bin"   # 本地权重
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================

# 加载CLIP模型和预处理
model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=PRETRAINED
)
model = model.visual.eval().to(DEVICE)

os.makedirs(FEAT_SAVE_ROOT, exist_ok=True)

def extract_video_features(video_path, frame_interval=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        count += 1
    cap.release()
    if len(frames) == 0:
        return None
    features = []
    with torch.no_grad():
        for i in range(0, len(frames), 32):
            batch = frames[i:i+32]
            batch_tensor = torch.stack([preprocess(f) for f in batch]).to(DEVICE)
            feats = model(batch_tensor).cpu().numpy()
            features.append(feats)
    features = np.concatenate(features, axis=0)  # [n, 512]
    return features

# 递归所有类别子文件夹和视频
for root, dirs, files in os.walk(DATA_ROOT):
    # 只处理视频文件（如mp4/avi）
    for vname in tqdm([f for f in files if f.lower().endswith(('.mp4', '.avi', '.mkv'))], desc=f"Extracting ({root})"):
        vpath = os.path.join(root, vname)
        feats = extract_video_features(vpath, frame_interval=FRAME_INTERVAL)
        if feats is not None:
            # 保持原有类别结构，特征文件名后缀加.npy
            rel_path = os.path.relpath(vpath, DATA_ROOT)
            save_path = os.path.join(FEAT_SAVE_ROOT, rel_path) + ".npy"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, feats)
        else:
            print(f"Warning: no frame extracted from {vname}")
