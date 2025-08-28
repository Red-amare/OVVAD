import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import math


# Temporal Adapter (TA) 模块
class TemporalAdapter(nn.Module):
    #对每一段视频，TA模块用时序邻接矩阵对每帧特征加权平均，补充CLIP没有的时序依赖
    def __init__(self, sigma=0.07, input_dim=512):
        super().__init__()
        self.sigma = sigma  # sigma控制相似度计算的敏感度
        self.ln = nn.LayerNorm(input_dim)  # 层归一化

    def forward(self, x):
        """
        x: 输入的视频特征 [B, T, C]
        B: batch size, T: video frame 数量, C: 特征维度
        """
        B, T, C = x.shape
        idxs = torch.arange(T, device=x.device)  # 获取帧的索引
        adj = torch.exp(-torch.abs(idxs[None, :] - idxs[:, None]) / self.sigma)  # 计算邻接矩阵
        adj = adj / adj.sum(dim=-1, keepdim=True)  # 归一化
        x = torch.matmul(adj, x)  # 用邻接矩阵加权输入特征
        x = self.ln(x)  # 层归一化
        return x

# SKI 模块：语义知识注入模块
class SKIModule(nn.Module):
    def __init__(self, clip_model, clip_processor, normal_prompts, abnormal_prompts):
        super().__init__()
        self.clip_model = clip_model  # CLIP模型用于获取文本特征
        self.clip_processor = clip_processor  # CLIP的文本处理器
        self.prompt_list = normal_prompts + abnormal_prompts  # 正常和异常类别的提示词
        self.prompt_num = len(self.prompt_list)
        # 通过 CLIP 提取语义嵌入
        self.semantic_emb = self.get_clip_emb(self.prompt_list)
        self.semantic_emb = nn.Parameter(self.semantic_emb, requires_grad=False)  # 固定参数

    def get_clip_emb(self, prompts):
        # prompts: List[str]
        inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(self.clip_model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.clip_model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def forward(self, xt):
        """
        xt: 视频帧特征 [B, T, C]
        计算视频特征与语义嵌入之间的相似度，并注入语义知识
        """
        sim = torch.sigmoid(torch.matmul(xt, self.semantic_emb.T))  # 计算视频特征与语义嵌入的相似度
        f_know = torch.matmul(sim, self.semantic_emb) / self.prompt_num  # 将相似度加权求平均
        x_ski = torch.cat([xt, f_know], dim=-1)  # 拼接视觉特征和语义知识
        return x_ski

# 只保留适配器功能（model.py里NASModule保持如下结构）
class NASModule(nn.Module):
    def __init__(self, feature_dim=512):
        super(NASModule, self).__init__()
        self.feature_dim = feature_dim  # 指定特征维度，通常和主模型视觉编码一致（如512）

    def forward(self, ext_feats, ext_prompts, novel_class_idx):
        # ext_feats: [N, T, C] 或 [N, C]，外部伪novel异常的特征（如XD片段特征）
        # ext_prompts: list[N]，每个特征对应的prompt文本（如"explosion"）
        # novel_class_idx: int 或 list[int]，novel类别编号（可全部同类，也可一批混合多类）

        device = ext_feats.device  # 获取输入特征所在设备（CPU或GPU）
        N = ext_feats.shape[0]  # 当前批次的样本数（XD片段数）

        # 生成异常标注（二分类标签），全为1，表示这些全是异常片段
        synthetic_labels = torch.ones(N, device=device)

        # 生成类别标签（多分类标签，指明每条伪异常的novel类别编号）
        if isinstance(novel_class_idx, int):
            # 如果只给了单一类别编号，则全部样本都属于这个novel类
            synthetic_class_labels = torch.full((N,), novel_class_idx, dtype=torch.long, device=device)
        else:
            # 如果给的是长度为N的类别编号list，则每条XD样本分配不同novel类别
            synthetic_class_labels = torch.tensor(novel_class_idx, dtype=torch.long, device=device)

        # prompt文本直接原样返回
        synthetic_prompts = ext_prompts

        # 返回：特征、异常二分类标签、novel类别标签、prompt文本
        # 可直接拼接到主训练batch，实现正常+base异常+NAS异常混合训练
        return ext_feats, synthetic_labels, synthetic_class_labels, synthetic_prompts

# class NASModule(nn.Module):
#     """
#     NAS (Novel Anomaly Synthesis)模块，用于将外部输入的伪novel异常特征和对应prompt整合为训练样本。
#     这里不做特征生成，仅起到“适配器”作用。
#     """
#     def __init__(self, feature_dim=512):
#         super(NASModule, self).__init__()
#         self.feature_dim = feature_dim
#
#     def forward(self, ext_feats, ext_prompts, novel_class_idx):
#         """
#         ext_feats: [N, T, C] tensor，外部输入（如XD异常片段特征）
#         ext_prompts: list[str]，每条特征对应的prompt（类别名称）
#         novel_class_idx: int 或 list[int]，novel类别的label编码
#         """
#         device = ext_feats.device
#         N = ext_feats.shape[0]
#         # 构造异常标签和类别标签
#         synthetic_labels = torch.ones(N, device=device)   # [N]，异常标签=1
#         if isinstance(novel_class_idx, int):
#             synthetic_class_labels = torch.full((N,), novel_class_idx, dtype=torch.long, device=device)
#         else:
#             synthetic_class_labels = torch.tensor(novel_class_idx, dtype=torch.long, device=device)
#         synthetic_prompts = ext_prompts
#         return ext_feats, synthetic_labels, synthetic_class_labels, synthetic_prompts



# 计算 Top-K 二元交叉熵损失
def topk_bce_loss(frame_logits, binary_labels, is_abnormal, k_ratio=1/16):

    '''
    异常视频：只取前K个最大分数（K=n/16），平均后与标签=1做BCE Loss
    正常视频：全帧均值（K=n），与标签=0做BCE Loss
    '''

    loss = 0
    B, T = frame_logits.shape
    for i in range(B):
        if is_abnormal[i]:
            k = max(1, int(T * k_ratio))  # 选择前 K 个最大分数
            score = frame_logits[i].topk(k).values.mean()  # 获取 Top-K 分数的平均值
        else:
            score = frame_logits[i].mean()  # 正常视频直接取平均值

        prob = torch.sigmoid(score)  # 先sigmoid
        # 这里就不能再用 with_logits 了，要用 binary_cross_entropy
        loss += F.binary_cross_entropy(prob.unsqueeze(0), binary_labels[i].unsqueeze(0))
    return loss / B

def topk_pooling(frame_scores, k=5):
    # frame_scores: shape [B, T] or [T]
    if frame_scores.ndim == 1:
        frame_scores = frame_scores.unsqueeze(0)
    B, T = frame_scores.shape
    pooled = []
    for i in range(B):
        k_ = min(k, T)
        pooled.append(frame_scores[i].topk(k_).values.mean())
    return torch.stack(pooled)

# 分类损失
def classification_loss(class_logits, class_labels):
    """
    计算分类损失
    只对有异常类别标签的样本（class_labels != -1）计算多类交叉熵损失
    """
    class_labels = class_labels.view(-1).long()  # 保证是一维 int64
    mask = (class_labels >= 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=class_logits.device)
    # 只有标签>=0才允许用作交叉熵
    return F.cross_entropy(class_logits[mask], class_labels[mask])

# SKI 模块相似度损失
def ski_sim_loss(video_feats, prompt_embs, class_labels, normal_idx, abnormal_idx, normal_class_idx, top_ratio=0.1):
    B, T, C = video_feats.shape
    loss = 0.0
    class_labels = class_labels.view(-1).long()  # 保证为一维 int64 tensor
    for i in range(B):
        k = max(1, int(T * top_ratio))
        v = F.normalize(video_feats[i], dim=-1)
        if class_labels[i].item() == normal_class_idx:
            sim = torch.matmul(v, prompt_embs[normal_idx].T)
        elif class_labels[i].item() >= 0 and class_labels[i].item() < prompt_embs.shape[0]:
            sim = torch.matmul(v, prompt_embs[abnormal_idx].T)
        else:
            continue
        sim_per_frame = sim.max(dim=1)[0]
        topk_val = sim_per_frame.topk(k).values.mean()
        loss += 1 - topk_val
    if B == 0:
        return torch.tensor(0.0, device=video_feats.device)
    return (loss / B).to(video_feats.device)

class OVVADModel(nn.Module):
    def __init__(self, clip_model, clip_processor, normal_prompts, abnormal_prompts, num_classes=13, use_ta=True,
                 use_ski=True):
        """
        Open-Vocabulary Video Anomaly Detection 主模型
        :param clip_model: 预训练CLIP模型（仅用于SKI模块文本特征提取）
        :param clip_processor: CLIP模型的文本处理器
        :param normal_prompts: 正常类别prompt列表
        :param abnormal_prompts: 异常类别prompt列表
        :param num_classes: 视频异常分类类别数（默认UCF-Crime为13）
        :param use_ta: 是否使用Temporal Adapter模块
        :param use_ski: 是否使用SKI语义知识注入模块
        """
        super().__init__()
        self.use_ta = use_ta
        self.use_ski = use_ski

        # Temporal Adapter（TA）：建模帧级时序依赖，提升检测泛化能力
        self.temporal_adapter = TemporalAdapter() if use_ta else nn.Identity()

        # SKI模块：注入CLIP文本prompt语义知识，拼接到视频帧特征
        self.ski_module = SKIModule(clip_model, clip_processor, normal_prompts,
                                    abnormal_prompts) if use_ski else nn.Identity()

        # 输入维度
        self.input_dim = 1024 if use_ski else 512

        # 帧级打分头（保留你原来的）
        hidden_dim = self.input_dim
        self.frame_scorer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

        # === 文本对齐式分类头：视频->512，对齐CLIP文本嵌入 ===
        self.fc_video_repr = nn.Linear(self.input_dim, 512)

        # 原来的线性分类器删掉（或注释掉）
        # self.classifier = nn.Linear(512, num_classes)

        # 类别文本嵌入（buffer，不参与训练）与可学习偏置（参与训练）
        self.register_buffer("class_text_emb", torch.zeros(num_classes, 512))  # [num_cls, 512]
        self.text_delta = nn.Parameter(torch.zeros(num_classes, 512))  # learnable per-class delta

        # 温度参数（类似CLIP的logit_scale）
        # 你可以初始化为 ln(14.285)=≈2.659（CLIP默认1/0.07），这里给个适中值
        self.logit_scale = nn.Parameter(torch.tensor(math.log(14.285)))

    @torch.no_grad()
    def set_class_text_emb(self, text_emb: torch.Tensor):
        """
        text_emb: [num_classes, 512]，来自CLIP文本编码器的类别嵌入
        """
        assert text_emb.shape == self.class_text_emb.shape
        # 确保是单位向量
        text_emb = F.normalize(text_emb, dim=-1)
        self.class_text_emb.copy_(text_emb)

    def forward(self, video_features):
        # 1) TA
        x_ta = self.temporal_adapter(video_features)           # [B, T, C]

        # 2) SKI
        x_ski = self.ski_module(x_ta)                          # [B, T, 2C] 或 [B, T, C]

        # 3) 帧级打分
        frame_logits = self.frame_scorer(x_ski).squeeze(-1)    # [B, T]

        # 4) 基于帧分数的attention聚合（建议对 x_ski 聚合，和文本更对齐）
        attn = torch.softmax(frame_logits, dim=1).unsqueeze(-1)
        video_repr = (x_ski * attn).sum(dim=1)                 # [B, input_dim]

        # 5) 文本对齐式分类
        vid_feat = self.fc_video_repr(video_repr)              # [B, 512]
        vid_feat = F.normalize(vid_feat, dim=-1)               # 归一化

        # 类别原型 = 冻结的文本嵌入 + 可学习delta（再归一化）
        cls_emb = self.class_text_emb + self.text_delta        # [num_cls, 512]
        cls_emb = F.normalize(cls_emb, dim=-1)

        # 相似度 * 温度
        logit_scale = self.logit_scale.exp()
        class_logits = logit_scale * (vid_feat @ cls_emb.T)    # [B, num_cls]

        return frame_logits, class_logits


    def forward_with_ski(self, x_ski, Ftext ):
        """
        x_ski: [B, T, 1024]  # 2C=1024
        Ftext: [l, C]         # 不直接用，仅保留
        """
        B, T, D = x_ski.shape
        # 1. 帧级异常分数
        frame_logits = self.frame_scorer(x_ski).squeeze(-1)  # [B, T]
        # 2. attention聚合得到视频级特征
        attn = torch.softmax(frame_logits, dim=1).unsqueeze(-1)  # [B, T, 1]
        video_repr = (x_ski * attn).sum(dim=1)  # [B, 1024]
        # 3. 分类头
        class_logits = self.classifier(video_repr)  # [B, num_classes]
        return frame_logits, class_logits

