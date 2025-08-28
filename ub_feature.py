import os
import torch
import clip
from PIL import Image
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

def extract_frames(video_path, sample_rate=16, max_frames=256):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count, saved = 0, 0
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            frames.append(img_pil)
            saved += 1
        count += 1
    cap.release()
    return frames

def extract_clip_features(frames, preprocess, model, device):
    feats_list = []
    batch_size = 16
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        imgs = [preprocess(frame).unsqueeze(0) for frame in batch_frames]
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(imgs)
            feats_list.append(feats.cpu().numpy())
    return np.concatenate(feats_list, axis=0)

def process_ubnormal(root_dir, save_dir, sample_rate=16):
    os.makedirs(save_dir, exist_ok=True)
    for scene in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        # 创建保存特征的 Scene 子文件夹
        scene_save_dir = os.path.join(save_dir, scene)
        os.makedirs(scene_save_dir, exist_ok=True)

        for fname in os.listdir(scene_path):
            if not fname.endswith(".mp4"):
                continue
            video_path = os.path.join(scene_path, fname)
            frames = extract_frames(video_path, sample_rate=sample_rate)
            if len(frames) == 0:
                print(f"Warning: no frames extracted from {video_path}")
                continue
            features = extract_clip_features(frames, preprocess, model, device)
            # 保存特征到 Scene 子文件夹
            save_name = f"{os.path.splitext(fname)[0]}.npy"
            save_path = os.path.join(scene_save_dir, save_name)
            np.save(save_path, features)
            print(f"Saved {save_path}, features shape: {features.shape}")

if __name__ == "__main__":
    dataset_root = "/data/UBnormal"
    save_dir = "/home/cxa/ub"
    process_ubnormal(dataset_root, save_dir, sample_rate=16)
