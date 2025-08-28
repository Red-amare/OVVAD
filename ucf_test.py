import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter
from transformers import CLIPModel, CLIPProcessor
from model import OVVADModel, topk_pooling
from collections import Counter

BASE_CLASSES = ['Abuse', 'Assault', 'Burglary', 'RoadAccidents', 'Robbery', 'Stealing']
NOVEL_CLASSES = ['Arrest', 'Arson', 'Explosion', 'Fighting', 'Shooting', 'Shoplifting', 'Vandalism']
ALL_CLASSES = BASE_CLASSES + NOVEL_CLASSES

class UCFClipFeatFolderDataset(Dataset):
    def __init__(self, feat_root, list_file, max_frames=256, class_names=ALL_CLASSES):
        self.samples = []
        self.labels = []
        self.max_frames = max_frames
        self.feat_root = feat_root
        self.class2idx = {c: i for i, c in enumerate(class_names)}

        with open(list_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            rel_path = line.strip()
            # 替换.mp4为.npy
            rel_path_npy = rel_path.replace('.mp4', '.npy')
            full_path = os.path.join(feat_root, rel_path_npy)
            if not os.path.exists(full_path):
                print(f"[Warning] Missing file: {full_path}")
                continue

            # 获取类别
            class_name = rel_path.split('/')[0]
            if 'Normal' in class_name:
                label = -1
            else:
                label = self.class2idx.get(class_name, None)
                if label is None:
                    print(f"[Warning] Unknown class: {class_name}")
                    continue

            self.samples.append(full_path)
            self.labels.append(label)

        print("最终样本统计：")
        print("正常样本数：", sum(1 for l in self.labels if l == -1))
        print("异常样本数：", sum(1 for l in self.labels if l != -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat = np.load(self.samples[idx])
        feat = torch.tensor(feat, dtype=torch.float32)
        label = self.labels[idx]
        T, C = feat.shape
        if T < self.max_frames:
            pad = torch.zeros(self.max_frames - T, C)
            feat = torch.cat([feat, pad], dim=0)
        else:
            feat = feat[:self.max_frames]
        binary_label = 0.0 if label == -1 else 1.0
        if label == -1:
            text_prompt = "normal activity"
            class_name = "Normal"
        else:
            class_name = list(self.class2idx.keys())[label]
            text_prompt = class_name.lower()
        return feat, text_prompt, torch.tensor(binary_label, dtype=torch.float32), class_name


def safe_roc_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    classes = set(y_true)
    if len(classes) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)

def safe_average_precision(y_true, y_score):
    from sklearn.metrics import average_precision_score
    classes = set(y_true)
    if len(classes) < 2:
        return float('nan')
    return average_precision_score(y_true, y_score)

def test_with_ski_prompt(model, test_loader, device, base_classes, novel_classes, topk=5):
    model.eval()
    all_labels, all_preds = [], []
    base_labels, base_preds = [], []
    novel_labels, novel_preds = [], []
    prompt_embs = model.ski_module.semantic_emb.to(device) if hasattr(model.ski_module, "semantic_emb") else None

    with torch.no_grad():
        for video_features, text_prompts, labels, label_strs in test_loader:
            video_features = video_features.to(device)
            labels = labels.to(device).float()
            frame_scores, _ = model(video_features, prompt_embs)
            probs = torch.sigmoid(frame_scores)
            video_scores = topk_pooling(probs, k=topk)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(video_scores.cpu().numpy())
            for i, cls_name in enumerate(label_strs):
                if cls_name == "Normal":
                    base_labels.append(labels[i].item())
                    base_preds.append(video_scores[i].item())
                    novel_labels.append(labels[i].item())
                    novel_preds.append(video_scores[i].item())
                elif cls_name in base_classes:
                    base_labels.append(labels[i].item())
                    base_preds.append(video_scores[i].item())
                elif cls_name in novel_classes:
                    novel_labels.append(labels[i].item())
                    novel_preds.append(video_scores[i].item())
                else:
                    print(f"Warning: Unknown class {cls_name} found, ignored in base/novel split.")

    print("==== UCF-Crime Detection Results (for Paper Table 1, SKI Prompt) ====")
    print("Label distributions:")
    print("Overall:", Counter(all_labels))
    print("Base:", Counter(base_labels))
    print("Novel:", Counter(novel_labels))
    overall_auc = safe_roc_auc(all_labels, all_preds)
    overall_ap = safe_average_precision(all_labels, all_preds)
    base_auc = safe_roc_auc(base_labels, base_preds)
    base_ap = safe_average_precision(base_labels, base_preds)
    novel_auc = safe_roc_auc(novel_labels, novel_preds)
    novel_ap = safe_average_precision(novel_labels, novel_preds)
    print(f"Overall  AUC: {overall_auc:.4f}  AP: {overall_ap:.4f}")
    print(f"Base     AUC: {base_auc:.4f}  AP: {base_ap:.4f}")
    print(f"Novel    AUC: {novel_auc:.4f}  AP: {novel_ap:.4f}")
    print("===================================")

def test_with_dummy_prompt(model, test_loader, device, base_classes, novel_classes, topk=5):
    model.eval()
    all_labels, all_preds = [], []
    base_labels, base_preds = [], []
    novel_labels, novel_preds = [], []
    num_class = model.classifier.out_features
    prompt_embs = torch.ones(num_class, 512, device=device)

    with torch.no_grad():
        for video_features, text_prompts, labels, label_strs in test_loader:
            video_features = video_features.to(device)
            labels = labels.to(device).float()
            frame_scores, _ = model(video_features, prompt_embs)
            probs = torch.sigmoid(frame_scores)
            video_scores = topk_pooling(probs, k=topk)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(video_scores.cpu().numpy())
            for i, cls_name in enumerate(label_strs):
                if cls_name == "Normal":
                    base_labels.append(labels[i].item())
                    base_preds.append(video_scores[i].item())
                    novel_labels.append(labels[i].item())
                    novel_preds.append(video_scores[i].item())
                elif cls_name in base_classes:
                    base_labels.append(labels[i].item())
                    base_preds.append(video_scores[i].item())
                elif cls_name in novel_classes:
                    novel_labels.append(labels[i].item())
                    novel_preds.append(video_scores[i].item())
                else:
                    print(f"Warning: Unknown class {cls_name} found, ignored in base/novel split.")

    print("==== UCF-Crime Detection Results (for Paper Table 1, Dummy Prompt) ====")
    print("Label distributions:")
    print("Overall:", Counter(all_labels))
    print("Base:", Counter(base_labels))
    print("Novel:", Counter(novel_labels))
    overall_auc = safe_roc_auc(all_labels, all_preds)
    overall_ap = safe_average_precision(all_labels, all_preds)
    base_auc = safe_roc_auc(base_labels, base_preds)
    base_ap = safe_average_precision(base_labels, base_preds)
    novel_auc = safe_roc_auc(novel_labels, novel_preds)
    novel_ap = safe_average_precision(novel_labels, novel_preds)
    print(f"Overall  AUC: {overall_auc:.4f}  AP: {overall_ap:.4f}")
    print(f"Base     AUC: {base_auc:.4f}  AP: {base_ap:.4f}")
    print(f"Novel    AUC: {novel_auc:.4f}  AP: {novel_ap:.4f}")
    print("===================================")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = CLIPModel.from_pretrained("/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3")  # 路径替换为你的本地路径
    clip_processor = CLIPProcessor.from_pretrained("/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3")
    clip_model = clip_model.to(device)

    normal_prompts = ['street', 'park', 'shopping mall', 'walking', 'running', 'working']
    abnormal_prompts = ['fighting', 'explosion', 'fire', 'robbery', 'shooting', 'arrest']

    model = OVVADModel(
        clip_model=clip_model,
        clip_processor=clip_processor,
        normal_prompts=normal_prompts,
        abnormal_prompts=abnormal_prompts,
        num_classes=len(ALL_CLASSES),
        use_ta=True,
        use_ski=False
    ).to(device)

    model.ski_module.clip_processor = clip_processor

    model.load_state_dict(torch.load('models/ucf_ovvad_final.pth', map_location=device))

    FEAT_ROOT = "/data/UCF_Crimes/Features/Video"   # 测试集特征主目录
    list_file = "/data/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt"  # 👈 替换成你的实际txt文件路径
    test_dataset = UCFClipFeatFolderDataset(FEAT_ROOT, list_file=list_file, max_frames=256)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 跑两个版本
    # test_with_ski_prompt(model, test_loader, device, BASE_CLASSES, NOVEL_CLASSES, topk=5)
    test_with_dummy_prompt(model, test_loader, device, BASE_CLASSES, NOVEL_CLASSES, topk=5)

if __name__ == "__main__":
    main()
