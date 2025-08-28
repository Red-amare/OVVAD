import torch
import numpy as np
import os

# XD异常特征片段合并

feat_dir = '/data/XD_features/'   # 这里写你的真实XD特征文件夹
feat_files = sorted([os.path.join(feat_dir, f) for f in os.listdir(feat_dir) if f.endswith('.npy')])

xd_feats_list = []
for f in feat_files:
    arr = np.load(f)           # [T, C]
    xd_feats_list.append(torch.tensor(arr, dtype=torch.float32))
xd_feats_all = torch.stack(xd_feats_list)  # [N, T, C]
torch.save(xd_feats_all, 'xd_feats.pt')