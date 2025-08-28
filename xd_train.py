import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import os
import re
from collections import defaultdict
from model import OVVADModel, NASModule, topk_bce_loss, classification_loss, ski_sim_loss
from transformers import CLIPModel, CLIPProcessor

# XD-Violence官方类别
XD_CLASSES = ['Abuse', 'Car accident', 'Explosion', 'Fighting', 'Riot', 'Shooting']

# 根据论文设定，排除多标签视频，只保留单标签
# XD-Base/Novel类别需参考论文（一般Fighting、Shooting等划为novel）
BASE_CLASSES = ['Fighting', 'Shooting', 'Car accident']
NOVEL_CLASSES = ['Abuse', 'Explosion', 'Riot']
ALL_CLASSES = ['Normal', 'Fighting', 'Shooting', 'Car accident', 'Abuse', 'Explosion', 'Riot']
class2idx = {c: i for i, c in enumerate(ALL_CLASSES)}

class XDClipFeatFolderDataset(Dataset):
    def __init__(self, feat_root, max_frames=256, class_names=None):
        self.samples = []
        self.labels = []
        self.max_frames = max_frames
        self.feat_root = feat_root

        if class_names is None:
            class_names = ALL_CLASSES
        self.class2idx = {c: i for i, c in enumerate(class_names)}

        npy_files = [f for f in os.listdir(self.feat_root) if f.endswith('.npy')]
        for fname in npy_files:
            full_path = os.path.join(self.feat_root, fname)
            m = re.search(r'label_([A-Z][0-9A-Z\-]*)', fname)
            if not m:
                print(f"[Warning] Unrecognized filename: {fname}")
                continue
            label_field = m.group(1)
            # 保留形如 B1-0-0 这种单主码（主码后全为0），但剔除B1-B2或B1,B2等真正多标签
            if ',' in label_field or re.search(r'B\d(-B\d)+', label_field):
                continue  # 逗号分割或B1-B2这类多标签
            # 剔除G-B2-0、B4-B1-G这种多主码
            if '-' in label_field:
                main_code = label_field.split('-')[0]
                # 后面如果都是0就是单标签
                if not all(x == '0' for x in label_field.split('-')[1:]):
                    continue
            else:
                main_code = label_field

            # ====== 这里先解析class_name ======
            if label_field == 'A':  # 这里就是 Normal
                class_name = 'Normal'
            elif label_field.startswith('B') or label_field.startswith('G'):
                B_code2name = {
                    'B1': 'Fighting',
                    'B2': 'Shooting',
                    'B4': 'Riot',
                    'B5': 'Abuse',
                    'B6': 'Car accident',
                    'G': 'Explosion'
                }
                main_code = label_field.split('-')[0]
                class_name = B_code2name.get(main_code, None)
                if class_name is None:
                    continue
            else:
                continue

            # ==== 关键修正：所有标签都落在 0 ~ len(ALL_CLASSES)-1 ====
            if class_name in self.class2idx:
                label = self.class2idx[class_name]
                self.samples.append(full_path)
                self.labels.append(label)

        normal_label = self.class2idx["Normal"]
        print("训练集样本数:", len(self.samples),
              "其中正常：", sum([1 for l in self.labels if l == normal_label]),
              "异常：", sum([1 for l in self.labels if l != normal_label]))
        base_class_counts = defaultdict(int)
        for label in self.labels:
            class_name = list(self.class2idx.keys())[label]
            if class_name in BASE_CLASSES:
                base_class_counts[class_name] += 1
        print("各base类异常样本数量：")
        for c in BASE_CLASSES:
            print(f"{c}: {base_class_counts[c]}")

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
        binary_label = 0.0 if label == self.class2idx["Normal"] else 1.0
        class_label = label
        class_name = list(self.class2idx.keys())[label]
        if class_name.lower() == "normal":
            text_prompt = "normal activity"
        else:
            text_prompt = class_name.lower()
        return feat, text_prompt, torch.tensor(binary_label, dtype=torch.float32), torch.tensor(class_label,
                                                                                                dtype=torch.long)

def get_balanced_sampler(dataset):
    import random
    normal_idx = dataset.class2idx["Normal"]
    normal_indices = [i for i, label in enumerate(dataset.labels) if label == normal_idx]
    abnormal_indices = [i for i, label in enumerate(dataset.labels) if label != normal_idx]
    min_len = min(len(normal_indices), len(abnormal_indices))
    if min_len == 0:
        raise ValueError("采样到的 normal 或 abnormal 数量为 0，无法训练！")
    normal_sample = random.sample(normal_indices, min_len)
    abnormal_sample = random.sample(abnormal_indices, min_len)
    indices = normal_sample + abnormal_sample
    random.shuffle(indices)
    print("采样后用于训练的样本数:", len(indices))
    return torch.utils.data.SubsetRandomSampler(indices)

# 训练一个epoch
def train_one_epoch(model, train_loader, optimizer, device,
                   Ftext, n_prompts):  # 只保留必要参数
    # ==== 提取normal和abnormal prompt列表，并获得其索引（用于loss等）====
    normal_prompts = model.ski_module.prompt_list[:len(model.ski_module.prompt_list) // 2]
    abnormal_prompts = model.ski_module.prompt_list[len(model.ski_module.prompt_list) // 2:]

    normal_idx = torch.tensor([ALL_CLASSES.index("Normal")], device=device)
    abnormal_idx = torch.tensor([i for i, c in enumerate(ALL_CLASSES) if c != "Normal"], device=device)
    normal_class_idx = ALL_CLASSES.index("Normal")

    model.train()
    total_loss = 0.0

    # ==== 主训练循环：每个batch ====
    for batch_idx, (video_feats, text_prompts, binary_labels, class_labels) in enumerate(train_loader):

        num_classes = len(ALL_CLASSES)  # 或你的类别总数

        # 转到cpu再numpy，方便调试
        class_labels_cpu = class_labels.detach().cpu().numpy()

        # 检查是否有非法值
        illegal = (class_labels_cpu < 0) | (class_labels_cpu >= num_classes)
        if illegal.any():
            print("非法class_labels batch内容:", class_labels_cpu)
            print("非法索引位置:", np.where(illegal)[0])
            raise ValueError(f"检测到非法class_labels: {class_labels_cpu[illegal]}")


        video_feats = video_feats.to(device)
        binary_labels = binary_labels.to(device)
        class_labels = class_labels.to(device)
        B, T, C = video_feats.shape

        # ==== 只用主数据集（无NAS拼接） ====
        feats_batch = video_feats
        labels_batch = binary_labels
        class_labels_batch = class_labels
        prompts_batch = list(text_prompts)

        # ==== SKI（语义知识注入）与前向传播 ====
        B = feats_batch.shape[0]
        xt = feats_batch.reshape(-1, C)
        sim = torch.sigmoid(torch.matmul(xt, Ftext.T))  # [B*T, n_prompts]
        Fknow = torch.matmul(sim, Ftext) / n_prompts
        x_ski = torch.cat([xt, Fknow], dim=-1).reshape(B, T, -1)
        frame_logits, class_logits = model.forward_with_ski(x_ski, Ftext)

        # ==== 损失计算 ====
        is_abnormal = (class_labels_batch != -1)
        loss_bce = topk_bce_loss(frame_logits, labels_batch, is_abnormal)
        loss_ce = classification_loss(class_logits, class_labels_batch)

        loss_ski = ski_sim_loss(feats_batch, Ftext, class_labels_batch, normal_idx, abnormal_idx, normal_class_idx)
        loss = loss_bce + loss_ce + loss_ski

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 5 == 0:
            print(f"Step [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch Average Loss: {avg_loss:.4f}")


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

    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)

    FEAT_ROOT = "/data/XD-Violence/Features/XDTrainClipFeatures"
    train_dataset = XDClipFeatFolderDataset(FEAT_ROOT, max_frames=256, class_names=ALL_CLASSES)

    epochs = 30
    batch_size = 64

    Ftext_prompts = [c.lower() for c in ALL_CLASSES]
    Ftext = model.ski_module.get_clip_emb(Ftext_prompts).to(device)
    n_prompts = len(Ftext_prompts)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        sampler = get_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, drop_last=True
        )
        # 调用只用主干模型的 train_one_epoch
        train_one_epoch(
            model, train_loader, optimizer, device,
            Ftext, n_prompts
        )

    torch.save(model.state_dict(), 'models/ovvad_xd.pth')


if __name__ == "__main__":
    main()
