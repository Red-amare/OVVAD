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

    @staticmethod
    def norm_name(name):
        import os, re
        base = os.path.basename(name).lower()
        base = re.sub(r'_x264', '', base)
        base = re.sub(r'\.(mp4|avi|mkv)$', '', base)
        return base

    def __init__(self, feat_root, list_file, max_frames=256, class_names=ALL_CLASSES,
                 anno_txt="Temporal_Anomaly_Annotation.txt",
                 video_root=None,             # << 新增：原始视频根目录（可选）
                 feat_stride=None):           # << 不再强依赖固定N；可留作回退
        """
        feat_stride: 一个特征向量代表多少原始帧。
          - 如果你的 .npy 是逐帧特征 -> feat_stride=1
          - 如果你的 .npy 是以 16 帧一个特征 -> feat_stride=16
        """
        self.samples = []          # 保存特征 .npy 的完整路径
        self.labels = []           # 视频级二分类：-1=Normal, 0..=异常类索引
        self.video_names = []      # 保存原始 mp4 文件名（如 'Arson011_x264.mp4'）
        self.max_frames = max_frames

        self.video_root = video_root
        self.frame_count_cache = {}          # << 缓存：mp4_name -> total_frames(int)
        self.feat_stride = feat_stride
        self.feat_root = feat_root
        self.feat_stride = feat_stride
        self.class2idx = {c: i for i, c in enumerate(class_names)}
        self.ann = load_temporal_annotations(anno_txt)  # 读入区间标注

        with open(list_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            rel_path_mp4 = line.strip()                  # 例如 'Arson/Arson011_x264.mp4'
            mp4_name = os.path.basename(rel_path_mp4)    # 'Arson011_x264.mp4'
            rel_path_npy = rel_path_mp4.replace('.mp4', '.npy')
            full_path = os.path.join(feat_root, rel_path_npy)
            if not os.path.exists(full_path):
                print(f"[Warning] Missing file: {full_path}")
                continue

            class_name = rel_path_mp4.split('/')[0]
            if 'Normal' in class_name:
                label = -1
            else:
                label = self.class2idx.get(class_name, None)
                if label is None:
                    print(f"[Warning] Unknown class: {class_name}")
                    continue

            self.samples.append(full_path)
            self.labels.append(label)
            self.video_names.append(mp4_name)

        print("最终样本统计：")
        print("正常样本数：", sum(1 for l in self.labels if l == -1))
        print("异常样本数：", sum(1 for l in self.labels if l != -1))

        # 1) 标注条数
        print("标注条数:", len(self.ann))

        # 2) 命中率（名称匹配）
        import re
        def norm_name(name):
            base = os.path.basename(name).lower()
            base = re.sub(r'_x264', '', base)
            base = re.sub(r'\.(mp4|avi|mkv)$', '', base)
            return base

        ann_keys_norm = {norm_name(k) for k in self.ann.keys()}
        hits = 0
        miss_samples = []
        for mp4 in self.video_names:
            if norm_name(mp4) in ann_keys_norm:
                hits += 1
            else:
                miss_samples.append(mp4)
        print(f"标注命中视频数: {hits}/{len(self.video_names)}")
        print("示例未命中:", miss_samples[:10])

        # 3) 正帧统计（抽样，避免太慢）
        pos = 0
        sample_n = min(len(self), 200)
        for i in range(sample_n):
            _, _, _, _, fl = self[i]  # 调用 __getitem__
            pos += int(fl.sum().item())
        print("抽样正帧总数:", pos)

        # 4) 抽查一个异常类视频的区间映射
        for i in range(len(self)):
            if self.labels[i] != -1:
                v = self.video_names[i]
                print("样例异常视频:", v, "标注区间(规范化):", self.ann.get(self.norm_name(v), "None"))
                print("样例异常视频(规范化)是否命中：", self.norm_name(v) in ann_keys_norm)
                break

    def __len__(self):
        return len(self.samples)

    def _get_total_frames(self, mp4_name):
        """读取/缓存原始视频总帧数；拿不到就返回 None。"""
        if mp4_name in self.frame_count_cache:
            return self.frame_count_cache[mp4_name]
        if self.video_root is None:
            return None
        # 通过 list_file 的相对路径可得到子目录
        # 这里假设 self.samples / self.video_names 对应 mp4_name = 'Class/Video_x264.mp4' 或只文件名
        # 你如果在 __init__ 里已经存了 rel_path_mp4，可以直接用它；这里用最保守的拼法：
        cand_paths = [
            os.path.join(self.video_root, mp4_name),                     # 仅文件名
        ]
        # 如果你保存了相对目录（推荐在 __init__ 里一并保存），可以加一条带子目录的候选路径
        # cand_paths.append(os.path.join(self.video_root, rel_dir, mp4_name))

        total = None
        for p in cand_paths:
            if os.path.exists(p):
                cap = cv2.VideoCapture(p)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                break
        if total is not None and total > 0:
            self.frame_count_cache[mp4_name] = total
        return total


    def _make_frame_labels(self, T, mp4_name):
        """
        用线性缩放把原始帧区间映射到特征索引。
        回退策略：如果拿不到 total_frames，则按旧的 feat_stride（N）去整除映射。
        """
        labels = np.zeros(T, dtype=np.float32)
        segs = self.ann.get(self.norm_name(mp4_name), [])

        if len(segs) == 0 or T <= 0:
            return labels

        total_frames = self._get_total_frames(mp4_name)

        if total_frames is not None and total_frames > 0:
            # === 线性缩放映射（推荐） ===
            # 标注常见是 1-based 且含端：先转 0-based
            for s, e in segs:
                s0 = max(0, s - 1)
                e0 = max(0, e - 1)
                # 映射到 [0, T-1]，用 floor；并确保闭区间
                fs = int(np.floor(s0 * T / total_frames))
                fe = int(np.floor(e0 * T / total_frames))
                fs = max(0, min(T - 1, fs))
                fe = max(fs, min(T - 1, fe))
                labels[fs:fe+1] = 1.0
            return labels
        else:
            # === 回退：还用固定步长 N 的闭区间“向下取整” ===
            if not self.feat_stride:
                # 没给 N：尝试估个近似N，避免严重漂移
                # 比如用一个保守整数：max(1, round(1.0 * total_frames / T))；但 total_frames=None 时只能用16作兜底
                N = 16
            else:
                N = int(self.feat_stride)
            for s, e in segs:
                s0 = max(0, s - 1); e0 = max(0, e - 1)
                fs = s0 // N
                fe = max(fs, e0 // N)
                fs = max(0, min(T - 1, fs))
                fe = max(fs, min(T - 1, fe))
                labels[fs:fe+1] = 1.0
            return labels

    def __getitem__(self, idx):
        feat = np.load(self.samples[idx])          # [T, C]
        feat = torch.tensor(feat, dtype=torch.float32)
        T, C = feat.shape

        mp4_name = self.video_names[idx]
        # 先按原始长度 T 生成帧标签（特征帧级）
        frame_labels_np = self._make_frame_labels(T, mp4_name)    # [T]
        frame_labels = torch.from_numpy(frame_labels_np)

        # 只在 T < max_frames 时做 padding；否则不做任何截断
        if self.max_frames is not None and T < self.max_frames:
            pad_len = self.max_frames - T
            feat = torch.cat([feat, torch.zeros(pad_len, C)], dim=0)
            frame_labels = torch.cat([frame_labels, torch.zeros(pad_len)], dim=0)

        # 二分类视频标签：是否有异常帧
        binary_label = 1.0 if frame_labels.sum().item() > 0 else 0.0

        # 文本提示与类别名（可选）
        label = self.labels[idx]
        if label == -1:
            class_name = "Normal"
            text_prompt = "normal activity"
        else:
            class_name = list(self.class2idx.keys())[label]
            text_prompt = class_name.lower()

        return feat, text_prompt, torch.tensor(binary_label, dtype=torch.float32), class_name, frame_labels

def load_temporal_annotations(txt_path):
    ann = {}
    import re, os
    def norm_name(name):
        base = os.path.basename(name).lower()
        base = re.sub(r'_x264', '', base)
        base = re.sub(r'\.(mp4|avi|mkv)$', '', base)
        return base

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            vid, cls_, s1, e1, s2, e2 = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            segs = []
            if s1 >= 0 and e1 >= 0:
                segs.append((s1, e1))
            if s2 >= 0 and e2 >= 0:
                segs.append((s2, e2))
            ann[norm_name(vid)] = segs
    return ann


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

def test_with_ski_prompt(model, test_loader, device, base_classes, novel_classes):
    """
    论文主文标准实现（frame-level评测）：
      - 每一帧都统计异常分数和帧标签
      - base组=normal所有帧+base异常帧
      - novel组=normal所有帧+novel异常帧
    """
    import numpy as np
    from collections import Counter

    model.eval()
    all_labels, all_scores = [], []
    base_labels, base_scores = [], []
    novel_labels, novel_scores = [], []

    # 获取SKI prompt embedding
    prompt_embs = model.ski_module.semantic_emb.to(device) if hasattr(model.ski_module, "semantic_emb") else None

    with torch.no_grad():
        for batch in test_loader:
            # 支持带/不带 text_prompts（不影响推理，只要label_strs和frame_labels在）
            if len(batch) == 5:
                video_features, text_prompts, labels, label_strs, frame_labels = batch
            else:
                video_features, labels, label_strs, frame_labels = batch
            video_features = video_features.to(device)
            frame_scores, _ = model(video_features)   # [B, T]
            probs = torch.sigmoid(frame_scores).cpu().numpy()      # [B, T]

            labels = labels.cpu().numpy()
            frame_labels = frame_labels.cpu().numpy()

            for i, cls_name in enumerate(label_strs):
                # 有效帧数（防止padding）
                valid_len = (np.abs(video_features[i].cpu().numpy()).sum(axis=1) != 0).sum()
                frame_score = probs[i, :valid_len]
                frame_label = frame_labels[i, :valid_len]

                # 汇总overall
                all_scores.extend(frame_score.tolist())
                all_labels.extend(frame_label.tolist())
                # 分组
                if cls_name == "Normal":
                    base_scores.extend(frame_score.tolist())
                    base_labels.extend(frame_label.tolist())
                    novel_scores.extend(frame_score.tolist())
                    novel_labels.extend(frame_label.tolist())
                elif cls_name in base_classes:
                    base_scores.extend(frame_score.tolist())
                    base_labels.extend(frame_label.tolist())
                elif cls_name in novel_classes:
                    novel_scores.extend(frame_score.tolist())
                    novel_labels.extend(frame_label.tolist())
                else:
                    print(f"Warning: Unknown class {cls_name}")

    # 打印帧数分布，便于对齐论文
    print("==== UCF-Crime Frame-level Results (SKI Prompt, Paper Standard) ====")
    print("Overall label dist:", Counter(all_labels))
    print("Base label dist:", Counter(base_labels))
    print("Novel label dist:", Counter(novel_labels))

    # 计算AUC与AP
    overall_auc = safe_roc_auc(all_labels, all_scores)
    overall_ap = safe_average_precision(all_labels, all_scores)
    base_auc = safe_roc_auc(base_labels, base_scores)
    base_ap = safe_average_precision(base_labels, base_scores)
    novel_auc = safe_roc_auc(novel_labels, novel_scores)
    novel_ap = safe_average_precision(novel_labels, novel_scores)

    print(f"Overall  AUC: {overall_auc:.4f}  AP: {overall_ap:.4f}")
    print(f"Base     AUC: {base_auc:.4f}  AP: {base_ap:.4f}")
    print(f"Novel    AUC: {novel_auc:.4f}  AP: {novel_ap:.4f}")
    print("===================================")

def test_with_dummy_prompt(model, test_loader, device, base_classes, novel_classes):
    """
    论文主文标准实现：
      - 每一帧都统计异常分数和帧标签
      - 统计所有帧的 ROC/AUC，分 overall/base/novel
    """
    model.eval()
    all_labels, all_scores = [], []
    base_labels, base_scores = [], []
    novel_labels, novel_scores = [], []

    num_class = model.classifier.out_features
    prompt_embs = torch.ones(num_class, 512, device=device)

    with torch.no_grad():
        for video_features, text_prompts, labels, label_strs, frame_labels in test_loader:
            # video_features: [B, T, C]
            # frame_labels:   [B, T]  # 帧标签（每一帧是0或1）

            video_features = video_features.to(device)
            frame_scores, _ = model(video_features, prompt_embs)  # [B, T]
            probs = torch.sigmoid(frame_scores)                   # [B, T]

            for i, cls_name in enumerate(label_strs):
                # 有效帧
                video_feat = video_features[i]
                valid_len = (video_feat.abs().sum(dim=1) != 0).sum().item()
                frame_score = probs[i, :valid_len].cpu().numpy()
                frame_label = frame_labels[i, :valid_len].cpu().numpy()   # [valid_len]

                # 汇总所有帧分数与帧标签
                all_scores.extend(frame_score.tolist())
                all_labels.extend(frame_label.tolist())

                # 分组
                if cls_name == "Normal":
                    # 全部normal帧加入base/novel
                    base_labels.extend(frame_label.tolist())
                    base_scores.extend(frame_score.tolist())
                    novel_labels.extend(frame_label.tolist())
                    novel_scores.extend(frame_score.tolist())
                elif cls_name in base_classes:
                    # 仅base类异常帧进base组
                    base_labels.extend(frame_label.tolist())
                    base_scores.extend(frame_score.tolist())
                elif cls_name in novel_classes:
                    # 仅novel类异常帧进novel组
                    novel_labels.extend(frame_label.tolist())
                    novel_scores.extend(frame_score.tolist())
                else:
                    print(f"Warning: Unknown class {cls_name}")

    print("==== UCF-Crime Frame-level Results (Paper Standard) ====")
    from collections import Counter
    print("Overall label dist:", Counter(all_labels))
    print("Base label dist:", Counter(base_labels))
    print("Novel label dist:", Counter(novel_labels))

    overall_auc = safe_roc_auc(all_labels, all_scores)
    base_auc = safe_roc_auc(base_labels, base_scores)
    novel_auc = safe_roc_auc(novel_labels, novel_scores)
    print(f"Overall  AUC: {overall_auc:.4f}")
    print(f"Base     AUC: {base_auc:.4f}")
    print(f"Novel    AUC: {novel_auc:.4f}")
    print("===================================")

    overall_auc = safe_roc_auc(all_labels, all_scores)
    overall_ap = safe_average_precision(all_labels, all_scores)
    base_auc = safe_roc_auc(base_labels, base_scores)
    base_ap = safe_average_precision(base_labels, base_scores)
    novel_auc = safe_roc_auc(novel_labels, novel_scores)
    novel_ap = safe_average_precision(novel_labels, novel_scores)
    print(f"Overall  AUC: {overall_auc:.4f}  AP: {overall_ap:.4f}")
    print(f"Base     AUC: {base_auc:.4f}  AP: {base_ap:.4f}")
    print(f"Novel    AUC: {novel_auc:.4f}  AP: {novel_ap:.4f}")
    print("===================================")



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = CLIPModel.from_pretrained("/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3")  # 路径替换为你的本地路径
    clip_processor = CLIPProcessor.from_pretrained("/home/cxa/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3")
    clip_model = clip_model.to(device)

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

    model.load_state_dict(torch.load('models/ucf_ovvad_final.pth', map_location=device))

    FEAT_ROOT = "/data/UCF_Crimes/Features/Video"   # 测试集特征主目录
    list_file = "/data/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt"  # 的实际txt文件路径

    # 推荐 batch_size=1（或用前面给过的 pad_collate 保证不截断）
    test_dataset = UCFClipFeatFolderDataset(
        FEAT_ROOT,
        list_file=list_file,
        max_frames=None,  # 或者给个很大的数；关键是不截断
        anno_txt="/data/UCF_Crimes/E_Features/Temporal_Anomaly_Annotation.txt",
        video_root="/data/UCF_Crimes/Videos",  # 你的原始视频根目录
        feat_stride=16  # 作为回退；线性缩放拿不到总帧数时才用
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 跑两个版本
    test_with_ski_prompt(model, test_loader, device, BASE_CLASSES, NOVEL_CLASSES)
    # test_with_dummy_prompt(model, test_loader, device, BASE_CLASSES, NOVEL_CLASSES)


if __name__ == "__main__":
    main()

