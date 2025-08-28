import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter
from transformers import CLIPModel, CLIPProcessor
from model import OVVADModel, topk_pooling

# XD-Violence 基础类和新颖类的划分
BASE_CLASSES = ['Fighting', 'Shooting', 'Car accident']
NOVEL_CLASSES = ['Abuse', 'Explosion', 'Riot']
ALL_CLASSES = ['Normal', 'Fighting', 'Shooting', 'Car accident', 'Abuse', 'Explosion', 'Riot']

# label mapping: class_name -> index
class2idx = {name: i for i, name in enumerate(ALL_CLASSES)}
base_class_idx = [class2idx[c] for c in BASE_CLASSES]
novel_class_idx = [class2idx[c] for c in NOVEL_CLASSES]

class XDClipFeatFolderDataset(Dataset):
    def __init__(self, feat_root, list_file, max_frames=256):
        self.samples = []
        self.labels = []
        self.frame_labels = []
        self.frame_label_classes = []
        self.max_frames = max_frames
        self.feat_root = feat_root

        normal_clip_count = 0
        abnormal_clip_count = 0

        # 用于全局片段统计
        total_normal_segments = 0
        total_abnormal_segments = 0

        with open(list_file) as f:
            for line in f:
                if not line.strip(): continue
                tokens = line.strip().split()
                rel_path = tokens[0]
                # ... 文件路径检查和读取略

                arr = np.load(full_path)
                T = min(arr.shape[0], max_frames)
                frame_label = np.zeros(max_frames, dtype=np.float32)
                frame_label_class = np.zeros(max_frames, dtype=np.int32) - 1

                # 处理异常段标注
                class_tokens = rel_path.split('label_')[-1].split('-')
                anomaly_intervals = []
                try:
                    for i in range(1, len(tokens), 2):
                        s, e = int(tokens[i]), int(tokens[i + 1])
                        s = max(0, min(s, max_frames - 1))
                        e = max(0, min(e, max_frames - 1))
                        frame_label[s:e + 1] = 1.0
                        # ...异常类别处理略
                        anomaly_intervals.append((s, e))
                except Exception as e:
                    print(f"[Error parsing line]: {line} => {e}")
                    continue

                self.samples.append(full_path)
                is_abnormal = frame_label.sum() > 0
                self.labels.append(1.0 if is_abnormal else 0.0)
                self.frame_labels.append(frame_label)
                self.frame_label_classes.append(frame_label_class)
                if is_abnormal:
                    abnormal_clip_count += 1
                else:
                    normal_clip_count += 1

                # ==== 修正片段统计 ====
                abnormal_seg = len(anomaly_intervals)
                # 合并区间、排序
                intervals = sorted(anomaly_intervals)
                normal_seg = 0
                prev_end = -1
                for s, e in intervals:
                    if s > prev_end + 1:
                        # 有正常区间
                        normal_seg += 1
                    prev_end = max(prev_end, e)
                if prev_end < T - 1:
                    normal_seg += 1  # 末尾剩余正常区间

                # 如果没有任何异常区间，整个clip就是1个正常区间
                if len(intervals) == 0:
                    normal_seg = 1

                total_abnormal_segments += abnormal_seg
                total_normal_segments += normal_seg

        print("最终样本统计：")
        print("clip级 正常样本段数：", normal_clip_count)
        print("clip级 异常样本段数：", abnormal_clip_count)
        all_labels_flat = np.concatenate(self.frame_labels)
        print("正常样本数（帧级）：", int((all_labels_flat == 0).sum()))
        print("异常样本数（帧级）：", int((all_labels_flat == 1).sum()))
        print("全体视频累计片段统计：")
        print("正常样本片段数：", total_normal_segments)
        print("异常样本片段数：", total_abnormal_segments)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat = np.load(self.samples[idx])
        feat = torch.tensor(feat, dtype=torch.float32)
        T, C = feat.shape
        if T < self.max_frames:
            pad = torch.zeros(self.max_frames - T, C)
            feat = torch.cat([feat, pad], dim=0)
        else:
            feat = feat[:self.max_frames]
        frame_labels = torch.tensor(self.frame_labels[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        frame_label_classes = torch.tensor(self.frame_label_classes[idx], dtype=torch.int32)
        return feat, label, frame_labels, frame_label_classes

def safe_average_precision(y_true, y_score):
    from sklearn.metrics import average_precision_score
    classes = set(y_true)
    if len(classes) < 2:
        return float('nan')
    return average_precision_score(y_true, y_score)

def test_with_ski_prompt(model, test_loader, device):
    model.eval()
    all_labels, all_scores = [], []
    all_base_labels, all_base_scores = [], []
    all_novel_labels, all_novel_scores = [], []
    prompt_embs = model.ski_module.semantic_emb.to(device) if hasattr(model.ski_module, "semantic_emb") else None

    with torch.no_grad():
        for batch in test_loader:
            video_features, labels, frame_labels, frame_label_classes = batch
            video_features = video_features.to(device)
            frame_scores, _ = model(video_features, prompt_embs)   # [B, T]
            probs = torch.sigmoid(frame_scores).cpu().numpy()
            frame_labels = frame_labels.cpu().numpy()
            frame_label_classes = frame_label_classes.cpu().numpy()
            for i in range(len(video_features)):
                valid_len = (np.abs(video_features[i].cpu().numpy()).sum(axis=1) != 0).sum()
                frame_score = probs[i, :valid_len]
                frame_label = frame_labels[i, :valid_len]
                frame_label_class = frame_label_classes[i, :valid_len]
                all_scores.extend(frame_score.tolist())
                all_labels.extend(frame_label.tolist())
                # base/novel 拆分
                for j in range(valid_len):
                    if frame_label[j] == 1:
                        if frame_label_class[j] in base_class_idx:
                            all_base_scores.append(frame_score[j])
                            all_base_labels.append(1)
                        elif frame_label_class[j] in novel_class_idx:
                            all_novel_scores.append(frame_score[j])
                            all_novel_labels.append(1)
                    else:
                        # 非异常帧，统一当作0加入
                        all_base_scores.append(frame_score[j])
                        all_base_labels.append(0)
                        all_novel_scores.append(frame_score[j])
                        all_novel_labels.append(0)

    from sklearn.metrics import average_precision_score
    print("==== XD-Violence Frame-level Results (SKI Prompt, Paper Standard) ====")
    print("Overall label dist:", Counter(all_labels))
    overall_ap = safe_average_precision(all_labels, all_scores)
    base_ap = safe_average_precision(all_base_labels, all_base_scores)
    novel_ap = safe_average_precision(all_novel_labels, all_novel_scores)
    print(f"Overall  AP: {overall_ap:.4f}")
    print(f"Base     AP: {base_ap:.4f}")
    print(f"Novel    AP: {novel_ap:.4f}")
    print("===================================")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = CLIPModel.from_pretrained("/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3")
    clip_processor = CLIPProcessor.from_pretrained("/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3")
    clip_model = clip_model.to(device)

    normal_prompts = ['street', 'park', 'shopping mall', 'walking', 'running', 'working']
    abnormal_prompts = ['Fighting', 'Explosion', 'Fire', 'Robbery', 'Shooting', 'Arrest', 'Fighting', 'Shooting', 'Car accident']
    model = OVVADModel(
        clip_model=clip_model,
        clip_processor=clip_processor,
        normal_prompts=normal_prompts,
        abnormal_prompts=abnormal_prompts,
        num_classes=len(ALL_CLASSES),
        use_ta=True,
        use_ski=True
    ).to(device)
    model.ski_module.clip_processor = clip_processor
    model.load_state_dict(torch.load('models/ovvad_xd.pth', map_location=device))
    FEAT_ROOT = "/home/cxa/xd/test"   # 测试集特征主目录
    list_file = "/home/cxa/xd/annotations.txt"
    test_dataset = XDClipFeatFolderDataset(FEAT_ROOT, list_file=list_file, max_frames=256)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_with_ski_prompt(model, test_loader, device)

if __name__ == "__main__":
    main()
