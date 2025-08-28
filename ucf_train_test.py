import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ------- 数据集 -------
class VideoFeatDataset_fromtxt(Dataset):
    def __init__(self, txt_file, feat_dir, class_names, label=1):
        # label=1: 异常（用 Anomaly_Train.txt），如果你有 NormalTrain.txt，label=0
        self.feat_dir = feat_dir
        self.class_names = class_names
        self.class_name2id = {name: i for i, name in enumerate(class_names)}
        self.label = label
        self.items = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                category, video_name = line.split('/')
                self.items.append((video_name, category))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        video_name, category = self.items[idx]
        # 载入帧特征
        feat_path = f"{self.feat_dir}/{video_name}.npy"
        feats = np.load(feat_path)
        feats = torch.from_numpy(feats).float()
        cat_id = self.class_name2id.get(category, -1)
        return feats, self.label, cat_id

# ------- 模型 -------
class OVVAD_Baseline(nn.Module):
    def __init__(self, feat_dim, class_names):
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.detector = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        # 分类器：只用帧均值做类别匹配
        self.cls_proj = nn.Linear(feat_dim, 512)  # 与文本特征对齐
        self.text_features = None  # 后续填充

    def set_text_features(self, text_features):
        self.text_features = text_features  # [num_classes, 512]

    def collate_fn(batch):
        # batch: list of (feats, label, cat_id)
        max_len = max(x[0].shape[0] for x in batch)
        feat_dim = batch[0][0].shape[1]
        batch_feats = []
        for x in batch:
            f = x[0]
            if f.shape[0] < max_len:
                pad = torch.zeros(max_len - f.shape[0], feat_dim)
                f = torch.cat([f, pad], dim=0)
            batch_feats.append(f)
        feats = torch.stack(batch_feats)
        labels = torch.tensor([x[1] for x in batch], dtype=torch.float)
        cat_ids = torch.tensor([x[2] for x in batch], dtype=torch.long)
        return feats, labels, cat_ids

    def forward(self, feats):
        # feats: (B, T, D)
        B, T, D = feats.shape
        # 1. 检测（逐帧）
        frame_logits = self.detector(feats)  # (B, T, 1)
        frame_scores = frame_logits.squeeze(-1)  # (B, T)
        # 2. 分类（视频级平均特征）
        video_feat = feats.mean(dim=1)  # (B, D)
        video_feat = self.cls_proj(video_feat)  # (B, 512)
        video_feat = video_feat / (video_feat.norm(dim=-1, keepdim=True) + 1e-6)
        if self.text_features is not None:
            logits_cls = video_feat @ self.text_features.T  # (B, num_classes)
        else:
            logits_cls = None
        return frame_scores, logits_cls

# ------- 获取文本特征 -------
def get_text_features(class_names, device):
    # 加载CLIP文本编码器
    CLIP_PATH = "/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
    clip_model = CLIPModel.from_pretrained(CLIP_PATH).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
    with torch.no_grad():
        inputs = clip_processor(text=class_names, return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**inputs)  # (num_classes, 512)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
    return text_features  # [num_classes, 512]

# ------- 训练与验证 -------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    total, loss_sum, acc_sum = 0, 0, 0
    for feats, labels, cat_ids in tqdm(loader):
        feats = feats.to(device)
        labels = labels.to(device)
        cat_ids = cat_ids.to(device)
        frame_scores, logits_cls = model(feats)
        # 视频级预测，取topK均值
        K = max(1, frame_scores.shape[1] // 16)
        pred_scores = torch.sigmoid(frame_scores.topk(K, dim=1).values.mean(dim=1))  # (B,)
        loss_det = bce(pred_scores, labels)
        # 只对异常做分类损失
        idx_abn = (labels == 1)
        if idx_abn.sum() > 0:
            loss_cls = ce(logits_cls[idx_abn], cat_ids[idx_abn])
        else:
            loss_cls = 0.0
        loss = loss_det + loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * feats.size(0)
        acc_sum += ((pred_scores > 0.5).float() == labels).sum().item()
        total += feats.size(0)
    print(f"Train Loss: {loss_sum/total:.4f}  Acc: {acc_sum/total:.4f}")

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total, loss_sum, acc_sum = 0, 0, 0
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    all_scores, all_labels = [], []
    for feats, labels, cat_ids in tqdm(loader):
        feats = feats.to(device)
        labels = labels.to(device)
        cat_ids = cat_ids.to(device)
        frame_scores, logits_cls = model(feats)
        K = max(1, frame_scores.shape[1] // 16)
        pred_scores = torch.sigmoid(frame_scores.topk(K, dim=1).values.mean(dim=1))
        loss_det = bce(pred_scores, labels)
        idx_abn = (labels == 1)
        if idx_abn.sum() > 0:
            loss_cls = ce(logits_cls[idx_abn], cat_ids[idx_abn])
        else:
            loss_cls = 0.0
        loss = loss_det + loss_cls
        loss_sum += loss.item() * feats.size(0)
        acc_sum += ((pred_scores > 0.5).float() == labels).sum().item()
        total += feats.size(0)
        all_scores.extend(pred_scores.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    print(f"Eval Loss: {loss_sum/total:.4f}  Acc: {acc_sum/total:.4f}")
    return all_scores, all_labels

# ------- 主流程 -------
def main():
    import os
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_names = ['Abuse', 'Arrest', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', 'Arson', 'RoadAccidents']
    train_txt = '/data/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt'
    feat_dir = '/data/UCF_Crimes/features'  # 所有 .npy 文件
    batch_size = 16
    epochs = 10

    # 文本特征
    text_features = get_text_features(class_names, device)

    # 用txt直接加载
    train_set = VideoFeatDataset_fromtxt(train_txt, feat_dir, class_names, label=1)

    # 如需添加正常视频，可再创建 normal_set 并合并
    # normal_set = VideoFeatDataset_fromtxt(normal_txt, feat_dir, class_names, label=0)
    # from torch.utils.data import ConcatDataset
    # train_set = ConcatDataset([train_set, normal_set])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # feat_dim 检查
    dummy_feat, _, _ = train_set[0]
    feat_dim = dummy_feat.shape[1]

    model = OVVAD_Baseline(feat_dim, class_names).to(device)
    model.set_text_features(text_features.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}")
        train_one_epoch(model, train_loader, optimizer, device)
        eval_model(model, val_loader, device)

if __name__ == '__main__':
    main()
