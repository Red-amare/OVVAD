import torch
from sympy import false
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from model import OVVADModel, NASModule, topk_bce_loss, classification_loss, ski_sim_loss
import os
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import SubsetRandomSampler
import re
from collections import Counter, defaultdict

LABEL_MAP = {
    1: ('B1', 'Arson', '纵火'),
    2: ('B2', 'Assault', '攻击 / 殴打'),
    3: ('B3', 'Burglary', '入室盗窃'),
    4: ('B4', 'Explosion', '爆炸'),
    5: ('B5', 'Fighting', '打架'),
    6: ('B6', 'RoadAccidents', '交通事故'),
}
ENGLISH_TYPE_TO_LABEL = {v[1]: k for k, v in LABEL_MAP.items()}   # 'Arrest': 0, ...
ENGLISH_TYPE_TO_CLASSNAME = {v[1]: v[1] for k, v in LABEL_MAP.items()}

# 定义基本类别和新类别
BASE_CLASSES = ['Abuse', 'Assault', 'Burglary', 'RoadAccidents', 'Robbery', 'Stealing']
NOVEL_CLASSES = ['Arrest', 'Arson', 'Explosion', 'Fighting', 'Shooting', 'Shoplifting', 'Vandalism']
ALL_CLASSES = BASE_CLASSES + NOVEL_CLASSES  # 所有类别合并
CLASS2IDX = {c: i for i, c in enumerate(ALL_CLASSES)}  # 类别与索引的映射关系

class UCFClipFeatFolderDataset(Dataset):
    def __init__(self, feat_root, list_file, max_frames=256, class_names=None):
        self.samples = []
        self.labels = []
        self.max_frames = max_frames
        self.feat_root = feat_root

        # 支持自定义类别顺序，否则默认按你原有 ALL_CLASSES
        if class_names is None:
            # 建议放在主程序中定义 ALL_CLASSES，然后传进来
            class_names = [
                'Abuse', 'Assault', 'Burglary', 'RoadAccidents', 'Robbery', 'Stealing',
                'Arrest', 'Arson', 'Explosion', 'Fighting', 'Shooting', 'Shoplifting', 'Vandalism'
            ]
        self.class2idx = {c: i for i, c in enumerate(class_names)}

        with open(list_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            rel_path = line.strip()
            rel_path_npy = rel_path.replace('.mp4', '.npy')
            full_path = os.path.join(feat_root, rel_path_npy)
            if not os.path.exists(full_path):
                print(f"[Warning] Missing file: {full_path}")
                continue

            # 类别
            class_name = rel_path.split('/')[0]
            if 'Normal' in class_name:
                label = -1
                self.samples.append(full_path)
                self.labels.append(label)
            elif class_name in BASE_CLASSES:
                label = self.class2idx[class_name]
                self.samples.append(full_path)
                self.labels.append(label)
            else:
                # 跳过novel类异常
                continue

        print("训练集样本数:", len(self.samples), "其中正常：", sum([1 for l in self.labels if l == -1]), "异常：", sum([1 for l in self.labels if l != -1]))

        base_class_counts = defaultdict(int)
        for i, label in enumerate(self.labels):
            # -1为Normal
            if label != -1:
                # 获取类别名
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
        binary_label = 0.0 if label == -1 else 1.0
        if label == -1:
            text_prompt = "normal activity"
            class_label = -1
        else:
            class_name = list(self.class2idx.keys())[label]
            text_prompt = class_name.lower()
            class_label = label
        return feat, text_prompt, torch.tensor(binary_label, dtype=torch.float32), torch.tensor(class_label, dtype=torch.long)


def get_balanced_sampler(dataset):
    """
    dataset.labels: [0/1/-1/...]
    -1 or 0为正常，其余为异常
    """
    import random
    # 兼容你的标签设定，假如正常是-1，其余是异常
    normal_indices = [i for i, label in enumerate(dataset.labels) if label in [-1, 0, 0.0]]
    abnormal_indices = [i for i, label in enumerate(dataset.labels) if label not in [-1, 0, 0.0]]
    min_len = min(len(normal_indices), len(abnormal_indices))
    # 随机采样，打乱
    normal_sample = random.sample(normal_indices, min_len)
    abnormal_sample = random.sample(abnormal_indices, min_len)
    indices = normal_sample + abnormal_sample
    random.shuffle(indices)
    return SubsetRandomSampler(indices)

# 训练一个epoch
def train_one_epoch(model, train_loader, optimizer, device, nas_module,
                   xd_feats_all, xd_prompts_all, xd_class_names_all, novel_class_map,
                   ski_text_emb,             # << 只传一份：SKI 提示词嵌入
                   nas_batch_size=8):
    # ==== 提取normal和abnormal prompt列表，并获得其索引（用于loss等）====
    normal_prompts = model.ski_module.prompt_list[:len(model.ski_module.prompt_list) // 2]
    abnormal_prompts = model.ski_module.prompt_list[len(model.ski_module.prompt_list) // 2:]
    normal_idx = torch.arange(len(normal_prompts), device=device)
    abnormal_idx = torch.arange(len(normal_prompts), len(normal_prompts) + len(abnormal_prompts), device=device)
    normal_class_idx = -1  # 依据数据集类别定义，通常normal设为-1或0

    model.train()           # 设置模型为训练模式
    total_loss = 0.0        # 累加所有batch的loss

    # ==== 主训练循环：每个batch ====
    for batch_idx, (video_feats, text_prompts, binary_labels, class_labels) in enumerate(train_loader):
        # video_feats: [B, T, C]，主数据集视频片段特征
        # text_prompts: list[B]，每个片段对应的prompt
        # binary_labels: [B]，二分类标签(0=正常, 1=异常)
        # class_labels: [B]，多分类标签(类别编号)

        video_feats = video_feats.to(device)
        binary_labels = binary_labels.to(device)
        class_labels = class_labels.to(device)
        B, T, C = video_feats.shape

        # ====== 1. 动态采样XD伪novel异常样本（即NAS模块输入）======
        if xd_feats_all.shape[0] >= nas_batch_size:
            # 如果NAS池数据充足，随机采样nas_batch_size个
            nas_indices = np.random.choice(xd_feats_all.shape[0], nas_batch_size, replace=False)
        else:
            # 否则就全用
            nas_indices = np.arange(xd_feats_all.shape[0])
        xd_feats_batch = xd_feats_all[nas_indices].to(device)   # [N, T, C]
        xd_prompts_batch = [xd_prompts_all[i] for i in nas_indices]
        # novel_class_map: 类别英文 -> 类别编号，如 'Explosion'->8
        xd_class_labels_batch = [novel_class_map[xd_class_names_all[i]] for i in nas_indices]

        # ====== 2. NAS模块生成一批伪novel异常（带标签和prompt）======
        nas_feats, nas_labels, nas_class_labels, nas_prompts = nas_module(
            xd_feats_batch, xd_prompts_batch, xd_class_labels_batch
        )
        # nas_feats: [N, T, C]
        # nas_labels: [N]，全为1
        # nas_class_labels: [N]，每条novel类别编号
        # nas_prompts: list[N]，prompt文本

        # ====== 3. 拼接主数据和NAS伪novel异常，形成一个训练大batch ======
        feats_batch = torch.cat([video_feats, nas_feats], dim=0)           # [B+N, T, C]
        labels_batch = torch.cat([binary_labels, nas_labels], dim=0)       # [B+N]
        class_labels_batch = torch.cat([class_labels, nas_class_labels], dim=0)   # [B+N]
        prompts_batch = list(text_prompts) + list(nas_prompts)             # list[B+N]


        # frame_logits, class_logits = model.forward_with_ski(x_ski, Ftext)   # 前向传播

        frame_logits, class_logits = model.forward(feats_batch)

        # ==== 5. 损失计算（异常检测+多分类+SKI语义） ====
        is_abnormal = (class_labels_batch != -1)             # 哪些是异常类
        loss_bce = topk_bce_loss(frame_logits, labels_batch, is_abnormal)        # 异常二分类损失
        loss_ce = classification_loss(class_logits, class_labels_batch)           # 多分类损失
        loss_ski = ski_sim_loss(feats_batch, ski_text_emb,
                                class_labels_batch, normal_idx, abnormal_idx, normal_class_idx)
        loss = loss_bce + loss_ce + loss_ski                 # 总loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 5 == 0:
            print(f"Step [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # === 6. 每个epoch结束，打印平均损失 ===
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch Average Loss: {avg_loss:.4f}")

# def train_one_epoch(model, train_loader, optimizer, device):
#     model.train()
#     total_loss = 0.0
#
#     for batch_idx, (video_feats, text_prompts, binary_labels, class_labels) in enumerate(train_loader):
#         video_feats = video_feats.to(device)
#         binary_labels = binary_labels.to(device)
#         class_labels = class_labels.to(device)
#
#         optimizer.zero_grad()
#
#         # 这里dummy prompt_embs只为接口兼容，实际forward时不会被用到
#         num_class = model.classifier.out_features
#         prompt_embs = torch.ones(num_class, 512, device=device)
#
#         # 前向传播
#         frame_logits, class_logits = model.forward(video_feats, prompt_embs)
#
#         # 只用二分类和分类损失，不计算SKI损失
#         # train_one_epoch内
#         is_abnormal = (class_labels != -1)
#         loss_bce = topk_bce_loss(frame_logits, binary_labels, is_abnormal)
#         loss_ce = classification_loss(class_logits, class_labels)
#         loss = loss_bce + loss_ce
#
#
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#         if (batch_idx + 1) % 5 == 0:
#             print(f"Step [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")
#
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch Average Loss: {avg_loss:.4f}")


# 主函数
def main():
    # 1. 设备选择（GPU优先，没有则用CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载CLIP主干模型和处理器（需替换为你的实际权重文件路径）
    clip_model = CLIPModel.from_pretrained("/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3")
    clip_processor = CLIPProcessor.from_pretrained("/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3")
    clip_model = clip_model.to(device)

    # 3. 定义normal和abnormal的prompt（按你的类别定义，可与ALL_CLASSES保持一致）
    normal_prompts = [    "street","sidewalk","crosswalk","alley","parking lot","gas station","shopping mall",
    "supermarket aisle","convenience store","checkout counter","office lobby","school hallway",
    "metro station","train platform","bus stop","park","plaza","playground","residential block",
    "apartment corridor","elevator lobby","stairwell","warehouse aisle","loading dock","campus quad",
    "cafeteria","hotel lobby","pharmacy","bank lobby","public square"]
    abnormal_prompts = [    "walking","standing","queueing","browsing shelves","window shopping","talking","calling on a phone",
    "texting","jogging","cycling","pushing a cart","carrying bags","entering","exiting","waiting",
    "cleaning","sweeping","mopping","stocking goods","delivering packages","checking out",
    "paying at cashier","taking an escalator","using an elevator","stretching","tying shoelaces",
    "adjusting a backpack","greeting","waving","sitting on a bench"]

    # 4. 初始化主模型，设置类别数、是否启用各模块
    model = OVVADModel(
        clip_model=clip_model,
        clip_processor=clip_processor,
        normal_prompts=normal_prompts,
        abnormal_prompts=abnormal_prompts,
        num_classes=len(ALL_CLASSES),
        use_ta=True,    # 是否使用 Temporal Adapter
        use_ski=True    # 是否使用 Semantic Knowledge Injection
    ).to(device)

    # 5. 初始化NAS模块，用于动态生成伪novel异常样本
    nas_module = NASModule(feature_dim=512).to(device)

    # 6. 优化器，包含主模型和NAS模块参数
    optimizer = torch.optim.Adam(list(model.parameters()) + list(nas_module.parameters()), lr=1e-4)

    # === UCF主训练集数据加载 ===
    FEAT_ROOT = "/data/UCF_Crimes/Features/Video"  #/data/UCF_Crimes/Features/Video
    TRAIN_TXT = "/data/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt"
    train_dataset = UCFClipFeatFolderDataset(
        FEAT_ROOT,
        list_file=TRAIN_TXT,
        max_frames=256,                  # 单个视频最大帧数，和特征对齐
        class_names=ALL_CLASSES
    )

    # === XD异常池加载：读取10个XD异常的npy特征，并对齐prompt和类别名 ===
    xd_feat_dir = './xd_nas/'
    feat_files = sorted([os.path.join(xd_feat_dir, f) for f in os.listdir(xd_feat_dir) if f.endswith('.npy')])

    max_frames = 256
    xd_feats_list = []
    xd_prompts_list = []
    xd_class_names_list = []
    for f in feat_files:
        arr = np.load(f)  # [T, 512]，单个XD异常片段特征
        T = arr.shape[0]
        if T < max_frames:
            # 帧数不足补0（右侧padding），保证所有XD片段shape一致
            pad = np.zeros((max_frames - T, arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:max_frames]
        xd_feats_list.append(torch.tensor(arr, dtype=torch.float32))
        # 解析异常类型英文名（如Explosion），后续与类别编号对齐
        m = re.search(r'label_(B\d)', f)
        if m:
            code = m.group(1)            # 例如 'B2'
            label_idx = int(code[1])
            eng_type = LABEL_MAP[label_idx][1]  # 英文类型名（如 'Explosion'）
        else:
            eng_type = 'Unknown'
        xd_prompts_list.append(eng_type.lower())   # 小写prompt文本
        xd_class_names_list.append(eng_type)       # 英文类别名

    # 汇总XD异常池到tensor和list
    xd_feats_all = torch.stack(xd_feats_list)      # [N, T, 512]
    xd_prompts_all = xd_prompts_list              # list[N]
    xd_class_names_all = xd_class_names_list      # list[N]

    # 定义novel类别到类别编号的映射（需和ALL_CLASSES一致）
    novel_class_map = {
        'Arrest': 6, 'Arson': 7, 'Explosion': 8, 'Fighting': 9,
        'Shooting': 10, 'Shoplifting': 11, 'Vandalism': 12
    }

    epochs = 20
    batch_size = 64

    # === A) 类别文本嵌入：仅用于“分类头对齐” ===
    class_text_emb = model.ski_module.get_clip_emb([c.lower() for c in ALL_CLASSES]).to(device)
    model.set_class_text_emb(class_text_emb)

    # === B) SKI 提示词嵌入：用于 SKI 注入 & ski_sim_loss ===
    ski_prompts = model.ski_module.prompt_list  # 初始化时= normal_prompts + abnormal_prompts
    ski_text_emb = model.ski_module.get_clip_emb(ski_prompts).to(device)

    # === 训练主循环 ===
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # 每轮都重新平衡采样，防止类别/标签分布不均
        sampler = get_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, drop_last=True
        )

        # 每个batch内动态注入NAS伪novel异常，训练整体loss
        train_one_epoch(
            model, train_loader, optimizer, device, nas_module,
            xd_feats_all, xd_prompts_all, xd_class_names_all, novel_class_map,
            ski_text_emb,  # << 用这份给 SKI
            nas_batch_size=8  # << 原来是 0，建议 >=8
        )

        # train_one_epoch(
        #     model, train_loader, optimizer, device
        # )

    # === 保存最终模型参数 ===
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/ucf_ovvad_final.pth')


if __name__ == '__main__':
    main()