import torch
from sympy import false
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
'''
TH's change
1 可学习温度因子，初始设为0.07
2 温度因子(可学习参数)删除,或者理解为固定为1不学习-------还行，但是没有本质区别，感觉就是一些误差
3 固定温度因子为0.07不学习-----------------------------效果不好
4 可学习温度因子，初始设为0.07，但是SKI模块最后变回512---效果不好，ski反而不如nothing了
'''
from model import OVVADModel, NASModule, topk_bce_loss, classification_loss, ski_sim_loss
# from model_new import OVVADModel, NASModule, topk_bce_loss, classification_loss, ski_sim_loss
import os
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import SubsetRandomSampler
import re
from collections import Counter, defaultdict
import pandas as pd
import sys
from xd_test import test_with_ski_prompt,XDClipFeatFolderDatasetTest
import random
from pathlib import Path
import csv

'''
TH's change
这部分是按照vadclip的标签重新排序了
'''
XD_LABEL_MAP = {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'}

# LABEL_MAP = {
#     1: ('arson', 'riot', '暴乱'),
#     2: ('assault', 'abuse', '攻击 / 殴打'),
#     3: ('explosion', 'explosion', '爆炸'),
# }
LABEL_MAP = {
    'arson':     ('riot', '暴乱'),
    'assault':   ('abuse', '攻击 / 殴打'),
    'explosion': ('explosion', '爆炸'),
}


# 定义基本类别和新类别
BASE_CLASSES = ['fighting','shooting','car accident']
NOVEL_CLASSES = ['riot', 'abuse','explosion']
ALL_CLASSES = BASE_CLASSES + NOVEL_CLASSES  # 所有类别合并
CLASS2IDX = {c: i for i, c in enumerate(ALL_CLASSES)}  # 类别与索引的映射关系

class XDClipFeatFolderDataset(Dataset):
    def __init__(self, list_file, max_frames=256, class_names=None):
        self.samples = []
        self.labels = []
        self.max_frames = max_frames

        # 支持自定义类别顺序，否则默认按你原有 ALL_CLASSES
        if class_names is None:
            # 建议放在主程序中定义 ALL_CLASSES，然后传进来
            class_names = [
                'fighting','shooting','car accident','riot', 'abuse','explosion'
            ]
        self.class2idx = {c: i for i, c in enumerate(class_names)}


        df = pd.read_csv(list_file, header=0)  # header=0表示第一行为表头，无表头可设为None

        all_path = df.iloc[:, 0]
        all_label = df.iloc[:, 1]
        for path, label in zip(all_path, all_label):
            if not os.path.exists(path):
                print(f"[Warning] Missing file: {path}")
                continue
            # 类别
            class_name = label[0] if label[0]=='G' or label[0]=='A' else label[:2]
            class_name = XD_LABEL_MAP[class_name]
            # print(f"class_name: {class_name}")
            if 'normal' in class_name:
                label = -1
                self.samples.append(path)
                self.labels.append(label)
            elif class_name in BASE_CLASSES:
                label = self.class2idx[class_name]
                self.samples.append(path)
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


def get_balanced_sampler(dataset):#取同样输数量的正常和异常样本
    """
    dataset.labels: [0/1/-1/...]
    -1 or 0为正常，其余为异常
    """
    import random
    # 兼容你的标签设定，假如正常是-1，其余是异常
    normal_indices = [i for i, label in enumerate(dataset.labels) if label in [-1, -1.0]]
    abnormal_indices = [i for i, label in enumerate(dataset.labels) if label not in [-1, -1.0]]
    min_len = min(len(normal_indices), len(abnormal_indices))
    # 随机采样，打乱
    normal_sample = random.sample(normal_indices, min_len)
    abnormal_sample = random.sample(abnormal_indices, min_len)
    indices = normal_sample + abnormal_sample
    random.shuffle(indices)
    return SubsetRandomSampler(indices)

def get_base_sampler(dataset):#提取异常样本，即训练集中的base异常
    """
    dataset.labels: [0/1/-1/...]
    -1 or 0为正常，其余为异常
    """
    # import random
    # 兼容你的标签设定，假如正常是-1，其余是异常
    abnormal_indices = [i for i, label in enumerate(dataset.labels) if label not in [-1,-1.0]]
    min_len = len(abnormal_indices)
    # 随机采样，打乱
    abnormal_sample = random.sample(abnormal_indices, min_len)
    indices = abnormal_sample
    # random.shuffle(indices)
    return SubsetRandomSampler(indices)


# 训练一个epoch
def train_one_epoch(model, train_loader, optimizer, device, nas_module,
                   ucf_feats_all, ucf_prompts_all, ucf_class_names_all, novel_class_map,
                   ski_text_emb,             # << 只传一份：SKI 提示词嵌入
                   nas_batch_size=8,use_nas=False,train_and_finetune_together=False):
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

        # ====== 1. 动态采样UCF伪novel异常样本（即NAS模块输入）======
        if ucf_feats_all.shape[0] >= nas_batch_size:
            # 如果NAS池数据充足，随机采样nas_batch_size个
            nas_indices = np.random.choice(ucf_feats_all.shape[0], nas_batch_size, replace=False)
        else:
            # 否则就全用
            nas_indices = np.arange(ucf_feats_all.shape[0])
        ucf_feats_batch = ucf_feats_all[nas_indices].to(device)   # [N, T, C]
        ucf_prompts_batch = [ucf_prompts_all[i] for i in nas_indices]
        # novel_class_map: 类别英文 -> 类别编号，如 'Explosion'->8
        ucf_class_labels_batch = [novel_class_map[ucf_class_names_all[i]] for i in nas_indices]

        # ====== 2. NAS模块生成一批伪novel异常（带标签和prompt）======
        nas_feats, nas_labels, nas_class_labels, nas_prompts = nas_module(
            ucf_feats_batch, ucf_prompts_batch, ucf_class_labels_batch
        )
        # nas_feats: [N, T, C]
        # nas_labels: [N]，全为1
        # nas_class_labels: [N]，每条novel类别编号
        # nas_prompts: list[N]，prompt文本

        '''
        TH's change
        选择是否拼接nas，同步训练和微调。
        '''
        if train_and_finetune_together:
            # ====== 拼接主数据和NAS伪novel异常，形成一个训练大batch ======
            feats_batch = torch.cat([video_feats, nas_feats], dim=0)           # [B+N, T, C]
            labels_batch = torch.cat([binary_labels, nas_labels], dim=0)       # [B+N]
            class_labels_batch = torch.cat([class_labels, nas_class_labels], dim=0)   # [B+N]
            prompts_batch = list(text_prompts) + list(nas_prompts)             # list[B+N]
        else:
            # ====== 不拼接nas，直接用原始数据进行训练 ====== 
            feats_batch = video_feats            # [B+N, T, C]
            labels_batch = binary_labels         # [B+N]
            class_labels_batch = class_labels    # [B+N]
            prompts_batch = list(text_prompts)   # list[B+N]


        # frame_logits, class_logits = model.forward_with_ski(x_ski, Ftext)   # 前向传播

        frame_logits, class_logits = model.forward(feats_batch)

        # ==== 5. 损失计算（异常检测+多分类+SKI语义） ====
        is_abnormal = (class_labels_batch != -1)             # 哪些是异常类
        loss_bce = topk_bce_loss(frame_logits, labels_batch, is_abnormal)        # 异常二分类损失
        loss_ce = classification_loss(class_logits, class_labels_batch)           # 多分类损失
        loss_ski = ski_sim_loss(feats_batch, ski_text_emb,
                                class_labels_batch, normal_idx, abnormal_idx, normal_class_idx)
        
        '''
        TH's change
        更改，选择性用nas模块
        如果不微调或者nas和主数据集一起训练，则这样计算loss
        '''
        if not use_nas or train_and_finetune_together:
            loss = loss_bce + loss_ce + loss_ski                 # 总loss

        else:
            #接下来计算novel部分的loss
            feats_batch_nas = nas_feats   
            labels_batch_nas =  nas_labels     
            class_labels_batch_nas = nas_class_labels
            prompts_batch_nas = list(nas_prompts)             # list[B+N]

            frame_logits_nas, class_logits_nas = model.forward(feats_batch_nas)

            is_abnormal_nas = (class_labels_batch_nas != -1)  
            loss_bce_nas = topk_bce_loss(frame_logits_nas, labels_batch_nas, is_abnormal_nas)        # 异常二分类损失
            loss_ce_nas = classification_loss(class_logits_nas, class_labels_batch_nas)           # 多分类损失

            loss = loss_bce_nas + loss_ce_nas + 1.0 *(loss_bce + loss_ce) #λ参数在ucf数据集设置为0.1，在xd设为1


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 5 == 0:
            print(f"Step [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # === 6. 每个epoch结束，打印平均损失 ===
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch Average Loss: {avg_loss:.4f}")

# 主函数
def main():
    # 1. 设备选择（GPU优先，没有则用CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip_model = CLIPModel.from_pretrained("./clip/clip_model")
    clip_processor = CLIPProcessor.from_pretrained("./clip/clip_processor")

    clip_model = clip_model.to(device)

    # 3. 定义normal和abnormal的prompt（按你的类别定义，可与ALL_CLASSES保持一致）
    '''
    第三版，各一百个，动名词混合   
    '''
    normal_prompts = [
    "street", "park", "office", "classroom", "kitchen", "supermarket", "shopping mall", "restaurant", "cafeteria", "library",
    "parking lot", "hospital", "corridor", "laboratory", "train station", "bus stop", "playground", "meeting room", "garden", "living room",
    "office desk", "shop", "lobby", "hallway", "crosswalk", "market", "coffee shop", "sports field", "warehouse", "subway station",
    "gym", "cinema", "home", "elevator", "store", "museum", "reception desk", "bridge", "stadium", "canteen",
    "corridor area", "computer room", "dining area", "public square", "campus", "office building", "bank", "bakery", "bus interior", "car interior",
    "walking", "running", "talking", "reading", "cooking", "cleaning", "studying", "resting", "driving", "parking",
    "eating", "drinking", "working", "chatting", "shopping", "typing", "relaxing", "exercising", "walking dog", "sitting",
    "standing", "waiting", "watching", "opening door", "closing door", "paying bill", "queuing", "crossing street", "answering phone", "checking phone",
    "taking photos", "folding clothes", "watering plants", "walking upstairs", "walking downstairs", "teaching", "attending meeting", "loading luggage", "unloading luggage", "walking together",
    "delivering package", "browsing shelves", "people gathering", "walking on sidewalk", "sitting quietly", "walking through hallway", "resting on chair", "people passing by"
    ]

    abnormal_prompts = [
    "fire", "explosion", "smoke", "blood", "weapon", "knife", "gun", "firelight", "firetruck", "police car",
    "ambulance", "fight scene", "robbery scene", "crash site", "burning vehicle", "destroyed building", "collapsed wall", "crowd chaos", "dangerous road", "car accident",
    "emergency area", "broken glass", "damaged shop", "street conflict", "riot", "panic crowd", "fallen person", "fire alarm", "street fire", "violent scene",
    "crime scene", "shooting area", "injured person", "vandalized area", "traffic collision", "explosion site", "crowd running", "gas leak", "smashed window", "wrecked car",
    "broken barrier", "robbery place", "fire zone", "shouting crowd", "gunfire sound", "debris", "alarm sound", "car fire", "collapsing structure", "screaming people",
    "dangerous place", "fighting", "stealing", "breaking glass", "shooting", "stabbing", "burning", "falling", "collapsing", "running away",
    "attacking", "chasing", "arguing", "punching", "kicking", "pushing", "escaping", "fainting", "bleeding", "vandalizing",
    "trespassing", "jumping fence", "breaking in", "destroying property", "lying on ground", "robbing", "overturning table", "setting fire", "kicking door", "breaking window",
    "shouting", "throwing objects", "pushing crowd", "smashing glass", "climbing wall", "reckless driving", "abnormal running", "jumping from height", "running across traffic", "throwing punches",
    "violent action", "panic behavior", "car overturn", "running toward danger", "grabbing bag", "breaking lock", "sudden explosion", "emergency event", "crowd panic", "dangerous behavior", "person injured"
    ]
    
    use_ta = True
    use_ski = False
    train_and_finetune_together = False
    lr = 1e-4
    name = ''
    if train_and_finetune_together:
        name = "TA_SKI_NAS"
    elif use_ta and use_ski:
        name = "TA_SKI"
    elif use_ta:
        name = "TA"
    elif use_ski:
        name = "SKI"
    else:
        name = "nothing"
    
    SEED=[42,42,42]

    random.seed(SEED[0])
    np.random.seed(SEED[1])
    torch.manual_seed(SEED[2])  # 固定种子，确保结果可复现
    
    # 4. 初始化主模型，设置类别数、是否启用各模块
    model = OVVADModel(
        clip_model=clip_model,
        clip_processor=clip_processor,
        normal_prompts=normal_prompts,
        abnormal_prompts=abnormal_prompts,
        num_classes=len(ALL_CLASSES),
        use_ta=use_ta,    # 是否使用 Temporal Adapter
        use_ski=use_ski   # 是否使用 Semantic Knowledge Injection
    ).to(device)

    # 5. 初始化NAS模块，用于动态生成伪novel异常样本
    nas_module = NASModule(feature_dim=512).to(device)

    # 6. 优化器，包含主模型和NAS模块参数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #训练阶段没有用到nas模块产生的异常样本，本质是nas也没有可训练参数

    # === XD主训练集数据加载 ===
    # TRAIN_CSV = r"./data/XD_Violence/Anomaly_Detection_splits/xd_CLIP_rgb.csv"
    TRAIN_CSV = r"./data/XD_Violence/Anomaly_Detection_splits/xd_CLIP_rgb_only_zero.csv"
    train_dataset = XDClipFeatFolderDataset(
        list_file=TRAIN_CSV,
        max_frames=256,                  # 单个视频最大帧数，和特征对齐
        class_names=ALL_CLASSES
    )

    # === UCF异常池加载：读取10个XD异常的npy特征，并对齐prompt和类别名 ===
    # list_file = r"./data/XD_Violence/Anomaly_Detection_splits/xd_CLIP_rgb_only_zero.csv"
    list_file = r"./data/UCF_Crimes/Anomaly_Detection_splits/ucf_CLIP_rgb_only_zero.csv"
    feat_files = []
    df = pd.read_csv(list_file, header=0)  # header=0表示第一行为表头，无表头可设为None

    all_path = df.iloc[:, 0]
    all_label = df.iloc[:, 1]
    for path, label in zip(all_path, all_label):
        if not os.path.exists(path):
            print(f"[Warning] Missing file: {path}")
            continue
        elif label == 'A' or label == 'Normal':
            continue
        feat_files.append(path)

    max_frames = 256
    ucf_feats_list = []
    ucf_prompts_list = []
    
    ucf_class_names_list = []
    for f in feat_files:
        arr = np.load(f)  # [T, 512]，单个XD异常片段特征
        T = arr.shape[0]
        if T < max_frames:
            # 帧数不足补0（右侧padding），保证所有XD片段shape一致
            pad = np.zeros((max_frames - T, arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:max_frames]
        '''
        TH's change
        按照新的label进行新的正则和映射
        并且保持一致，只取__1.npy的文件
        '''
        # 匹配两种格式：
        # 1. label_B数字（如 label_B1、label_B2）
        # 2. label_G（单独的G，无数字）

        # 1. 提取文件名（排除路径，只取文件名部分）
        filename = os.path.basename(f)  # 例如从"/xxx/abc__1.npy"中提取"abc__1.npy"
        # 2. 判断文件名是否以 ".npy" 结尾
        if filename.endswith("__0.npy"):
            m_xd = re.search(r'label_(B\d|G)', filename)  # 核心：用 | 分隔两个分支
            m_ucf = re.search(r'^[A-Za-z]+(?=\d)', filename)    
            if m_xd:
                code = m_xd.group(1)  # 提取匹配到的内容（如 'B3' 或 'G'） 
                eng_type = XD_LABEL_MAP[code]
            elif m_ucf:
                code = m_ucf.group(0)
                code = code.lower()
                if code in LABEL_MAP.keys():
                    eng_type = LABEL_MAP[code][0]
                else:#base类异常直接不要
                    eng_type = 'Unknown'
            else:
                eng_type = 'Unknown'
                print(f"未匹配到ucf或xd数据集: {filename}")
            # print(f"eng_type: {eng_type}")
            '''
            TH's change.
            只将ucf中属于xd的novel类别的异常加入集合
            但实际ucf_nas我已经处理好了，全是novel
            '''
            if eng_type in NOVEL_CLASSES:
                ucf_prompts_list.append(eng_type.lower())   # 小写prompt文本
                ucf_class_names_list.append(eng_type)       # 英文类别名
                ucf_feats_list.append(torch.tensor(arr, dtype=torch.float32))

    # 汇总XD异常池到tensor和list
    ucf_feats_all = torch.stack(ucf_feats_list)      # [N, T, 512]
    ucf_prompts_all = ucf_prompts_list              # list[N]
    ucf_class_names_all = ucf_class_names_list      # list[N]

    # 定义novel类别到类别编号的映射（需和ALL_CLASSES一致）
    novel_class_map = {
        'riot':3, 'abuse':4,'explosion':5
    }

    epochs = 20
    batch_size = 64

    # === A) 类别文本嵌入：仅用于“分类头对齐” ===
    class_text_emb = model.ski_module.get_clip_emb([c.lower() for c in ALL_CLASSES]).to(device)
    model.set_class_text_emb(class_text_emb)

    # === B) SKI 提示词嵌入：用于 SKI 注入 & ski_sim_loss ===
    ski_prompts = model.ski_module.prompt_list  # 初始化时= normal_prompts + abnormal_prompts
    ski_text_emb = model.ski_module.get_clip_emb(ski_prompts).to(device)


    #加载测试集
    test_list_file = r"./data/XD_Violence/Anomaly_Detection_splits/xd_CLIP_rgbtest.csv"  # 的实际txt文件路径

    # 推荐 batch_size=1（或用前面给过的 pad_collate 保证不截断）
    test_dataset = XDClipFeatFolderDatasetTest(
        list_file=test_list_file,
        max_frames=None,  # 或者给个很大的数；关键是不截断
        anno_txt=r"./data/XD_Violence/E_Features/annotations.txt",
        # video_root="/data/UCF_Crimes/Videos",  # 你的原始视频根目录
        feat_stride=16  # 作为回退；线性缩放拿不到总帧数时才用
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

     #记录checkpoint
    csv_filename = f"./checkpoint/XD_Ehanced_{name}_{SEED}_lr-{lr}.csv"
    csv_path = Path(csv_filename)  # 确保路径正确
    if csv_path.exists():
        # 拆分文件名和扩展名，添加_copy后重新组合
        csv_path = csv_path.with_name(f"{csv_path.stem}_copy{csv_path.suffix}")
    
    # 确保父目录存在
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 表头顺序：epoch + 所有指标
        writer.writerow([
            "epoch", 
            "overall_auc", "base_auc", "novel_auc",
            "overall_ap", "base_ap", "novel_ap"
        ])

    # === 训练主循环 ===
    best_ap = 0
    for epoch in range(1, epochs + 1):
        print(f"\ntrain_Epoch {epoch}/{epochs}")

        '''
        TH's change
        每轮都重新平衡采样，防止类别/标签分布不均
        但这样会浪费一些正样本，因为正样本比负样本多
        或者
        sampler=None,shuffle=True
        利用完所有的正样本
        '''
        sampler = get_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            sampler=sampler,
            num_workers=4, drop_last=True,
            # shuffle=True
        )

        # 训练阶段不加入NAS，训练整体loss
        train_one_epoch(
            model, train_loader, optimizer, device, nas_module,
            ucf_feats_all, ucf_prompts_all, ucf_class_names_all, novel_class_map,
            ski_text_emb,  # << 用这份给 SKI
            nas_batch_size=32,  # << 原来是 0，建议 >=8
            use_nas=False,
            train_and_finetune_together=train_and_finetune_together
        )

        # train_one_epoch(
        #     model, train_loader, optimizer, device
        # )
        overall_auc, base_auc, novel_auc, overall_ap, base_ap, novel_ap = test_with_ski_prompt(model, test_loader, device, BASE_CLASSES, NOVEL_CLASSES)
        if overall_ap > best_ap:
            best_ap = overall_ap
            # === 保存最终训练模型参数 ===
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/xd_ovvad_train_{name}.pth')

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(overall_auc, 4),  # 保留4位小数
                round(base_auc, 4),
                round(novel_auc, 4),
                round(overall_ap, 4),
                round(base_ap, 4),
                round(novel_ap, 4)
            ])
    print(f"best overall AP: {best_ap:.4f}")

    '''
    TH's change
    添加NAS微调
    '''
    # if train_and_finetune_together == True:
    #     sys.exit()

    # #== NAS微调主循环 ==
    # #== 加载model_finetune方式二选一，一个是一路训练下来，一个是从本地加载 ==
    # model_finetune = model.to(device)
    # model_finetune.load_state_dict(torch.load(f'models/xd_ovvad_train{name}.pth'))#可以接着微调,注意，由于model不一定是最优的，所以要加载

    # finetune_epoches = 10
    # finetune_batch_size = 10
    # # optimizer_finetune = torch.optim.Adam(list(model_finetune.parameters()) + list(nas_module.parameters()), lr=5e-6)
    # optimizer_finetune = torch.optim.Adam(model_finetune.parameters(), lr=5e-6)#nas没有可训练参数，本质没区别

    # best_finetune_ap = 0
    # for epoch in range(1, finetune_epoches + 1):
    #     print(f"\nfinetune_Epoch {epoch}/{finetune_epoches}")
    #     base_sampler = get_base_sampler(train_dataset)#获取base异常
    #     # base_sampler = get_balanced_sampler(train_dataset)#获取正常数据集，可以试一下是不是加了正常数据集微调就可以正确。
    #     loader_finetune = DataLoader(
    #         train_dataset, batch_size=finetune_batch_size, 
    #         sampler=base_sampler,
    #         num_workers=4, drop_last=True
    #     )
    #     #每个batch内动态注入NAS伪novel异常
    #     train_one_epoch(
    #         model_finetune, loader_finetune, optimizer_finetune, device, nas_module,
    #         ucf_feats_all, ucf_prompts_all, ucf_class_names_all, novel_class_map,
    #         ski_text_emb,  # << 用这份给 SKI
    #         nas_batch_size=finetune_batch_size,  # 每批次取10个base和10个nas
    #         use_nas=True,
    #         train_and_finetune_together=train_and_finetune_together
    #     )

    #     overall_auc,base_auc,novel_auc,overall_ap,base_ap,novel_ap = test_with_ski_prompt(model, test_loader, device, BASE_CLASSES, NOVEL_CLASSES)
    #     if overall_ap > best_finetune_ap:
    #         best_finetune_ap = overall_ap
    #         # === 保存最终训练模型参数 ===
    #         os.makedirs('models', exist_ok=True)
    #         torch.save(model_finetune.state_dict(), f'models/xd_ovvad_finetune_{name}.pth')



if __name__ == '__main__':
    main()