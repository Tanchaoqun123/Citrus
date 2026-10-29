"""
dataset_seq.py
Stage2 数据集：目标值改为完整时序序列 [T]，而非单点值
Stage1 数据集与原版本完全相同，直接复用逻辑
"""

import os
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# 时间点配置
# ---------------------------------------------------------------------------

_TIME_POINTS = [3, 6, 9, 12, 15, 20, 25, 35, 45]

_IMG_FOLDER_MAP = {
    3:  '3天',  6:  '9天',
    9:  '9天',  12: '15天',
    15: '15天', 20: '20天',
    25: '25天', 35: '35天',
    45: '45天',
}


# ---------------------------------------------------------------------------
# Stage 1 数据集（与原版本相同）
# ---------------------------------------------------------------------------

class Stage1Dataset(Dataset):
    """第一阶段数据集：纯图像 + 标签（成熟度 / 新鲜度等级）"""

    _MATURITY_FOLDERS = {'qingpi': 0, 'erhong': 1, 'dahong': 2}
    _FRESHNESS_FOLDERS = {'0d': 0, '7d': 1, '14d': 2, '21d': 3,
                          '28d': 4, '35d': 5, '42d': 6, '49d': 7}

    def __init__(self, data_dir, transform=None, is_train=True, val_ratio=0.2, seed=42):
        self.transform = transform

        maturity_root  = os.path.join(data_dir, 'images_stage1', 'maturity_chashugan')
        freshness_root = os.path.join(data_dir, 'images_stage1', 'freshness_wenzhoumigan')

        samples = []

        for folder, label in self._MATURITY_FOLDERS.items():
            folder_path = os.path.join(maturity_root, folder)
            for p in sorted(glob.glob(os.path.join(folder_path, '*.JPG')) +
                            glob.glob(os.path.join(folder_path, '*.jpg'))):
                samples.append((p, label, -1))

        for folder, label in self._FRESHNESS_FOLDERS.items():
            folder_path = os.path.join(freshness_root, folder)
            for p in sorted(glob.glob(os.path.join(folder_path, '*.JPG')) +
                            glob.glob(os.path.join(folder_path, '*.jpg'))):
                samples.append((p, -1, label))

        rng     = np.random.default_rng(seed)
        indices = rng.permutation(len(samples))
        split   = int(len(samples) * (1 - val_ratio))
        chosen  = indices[:split] if is_train else indices[split:]
        self.samples = [samples[i] for i in chosen]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, maturity, freshness = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except OSError:
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)

        return {
            'image':    img,
            'maturity': torch.tensor(maturity,  dtype=torch.long),
            'freshness':torch.tensor(freshness, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Excel 读取工具
# ---------------------------------------------------------------------------

def _load_excel_series(path, col_idx):
    """读取 Excel 中指定列的 9 个时间点数值，自动跳过非数值行。"""
    df     = pd.read_excel(path, header=None)
    series = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna()
    values = series.values
    assert len(values) == 9, f"{path} 期望9个数值，实际{len(values)}个"
    return values.astype(np.float32)


def _load_dianzibi(path, col_idx):
    """读取电子鼻 Excel，返回 (9, 8) 数组。"""
    df = pd.read_excel(path, header=None)
    if col_idx == 0:
        block = df.iloc[3:12, 1:9]
    else:
        block = df.iloc[15:24, 1:9]
    return block.astype(float).values.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 2 数据集（序列版本）
# ---------------------------------------------------------------------------

class Stage2SeqDataset(Dataset):
    """第二阶段数据集（序列预测版本）

    与原版本的唯一区别：
      原版本：目标值取最后一个时间点的单个标量
      本版本：目标值为完整时序序列 [T]，提供 T 倍监督信号

    返回字段：
      img_seq      : [T, 3, H, W]
      et_seq       : [T, 1, 8]
      enz_seq      : [T]
      wl_seq       : [T]
      fcr_seq      : [T]   ← 完整腐败率序列
      moisture_seq : [T]   ← 完整水分序列
      hardness_seq : [T]   ← 完整硬度序列
      maturity     : scalar (-1)
      freshness    : scalar (-1.0)
      storage_cond : scalar
    """

    def __init__(self, data_dir, config, is_train=True, transform=None):
        self.config    = config
        self.transform = transform
        self.T         = config.num_time_steps
        self.img_size  = config.img_size

        stage2_root = os.path.join(data_dir, 'baoxian_stage2')

        dianzibi_path    = os.path.join(stage2_root, 'dianzibi.xlsx')
        meihuoxing_path  = os.path.join(stage2_root, 'meihuoxing.xlsx')
        shizhonglv_path  = os.path.join(stage2_root, 'shizhonglv.xlsx')
        shuifen_path     = os.path.join(stage2_root, 'shuifen.xlsx')
        fubailv_path     = os.path.join(stage2_root, 'fubailv.xlsx')
        yingdu_path      = os.path.join(stage2_root, 'yingdu.xlsx')

        self.et_data      = [_load_dianzibi(dianzibi_path, c)              for c in range(2)]
        self.enz_data     = [_load_excel_series(meihuoxing_path, c + 1)    for c in range(2)]
        self.wl_data      = [_load_excel_series(shizhonglv_path, c + 1)    for c in range(2)]
        self.moist_data   = [_load_excel_series(shuifen_path,    c + 1)    for c in range(2)]
        self.fcr_data     = [_load_excel_series(fubailv_path,    c + 1)    for c in range(2)]
        self.yingdu_data  = [_load_excel_series(yingdu_path,     c + 1)    for c in range(2)]

        cond_folders = {0: 'changwen', 1: 'lengcang'}
        self.img_paths = {}
        for cond, folder in cond_folders.items():
            folder_path = os.path.join(stage2_root, folder)
            imgs = {}
            for p in sorted(glob.glob(os.path.join(folder_path, '*.JPG')) +
                            glob.glob(os.path.join(folder_path, '*.jpg'))):
                name = os.path.splitext(os.path.basename(p))[0]
                imgs[name] = p
            self.img_paths[cond] = imgs

        self.conditions = [0, 1]

    def __len__(self):
        return len(self.conditions)

    def _load_img_seq(self, cond):
        imgs   = self.img_paths[cond]
        frames = []
        for t in _TIME_POINTS:
            folder_name = _IMG_FOLDER_MAP[t]
            img_path    = imgs[folder_name]
            img         = Image.open(img_path).convert('RGB')
            if self.transform:
                frame = self.transform(img)
            else:
                img   = img.resize((self.img_size, self.img_size))
                frame = torch.from_numpy(
                    np.array(img, dtype=np.float32) / 255.0
                ).permute(2, 0, 1)
            frames.append(frame)
        return torch.stack(frames)   # [T, 3, H, W]

    def __getitem__(self, idx):
        cond = self.conditions[idx]

        img_seq = self._load_img_seq(cond)                          # [T, 3, H, W]
        et_seq  = torch.from_numpy(self.et_data[cond]).unsqueeze(1) # [T, 1, 8]
        enz_seq = torch.from_numpy(self.enz_data[cond])             # [T]
        wl_seq  = torch.from_numpy(self.wl_data[cond])              # [T]

        # ---- 核心改动：目标值为完整序列 [T] ----
        fcr_seq      = torch.from_numpy(self.fcr_data[cond])        # [T]
        moisture_seq = torch.from_numpy(self.moist_data[cond])      # [T]
        hardness_seq = torch.from_numpy(self.yingdu_data[cond])     # [T]

        return {
            'img_seq':      img_seq,
            'et_seq':       et_seq,
            'enz_seq':      enz_seq,
            'wl_seq':       wl_seq,
            'fcr_seq':      fcr_seq,
            'moisture_seq': moisture_seq,
            'hardness_seq': hardness_seq,
            'maturity':     torch.tensor(-1,   dtype=torch.long),
            'freshness':    torch.tensor(-1.0, dtype=torch.float32),
            'storage_cond': torch.tensor(cond, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'img_seq':      torch.stack([b['img_seq']      for b in batch]),
            'et_seq':       torch.stack([b['et_seq']       for b in batch]),
            'enz_seq':      torch.stack([b['enz_seq']      for b in batch]),
            'wl_seq':       torch.stack([b['wl_seq']       for b in batch]),
            'fcr_seq':      torch.stack([b['fcr_seq']      for b in batch]),
            'moisture_seq': torch.stack([b['moisture_seq'] for b in batch]),
            'hardness_seq': torch.stack([b['hardness_seq'] for b in batch]),
            'maturity':     torch.stack([b['maturity']     for b in batch]),
            'freshness':    torch.stack([b['freshness']    for b in batch]),
            'storage_cond': torch.stack([b['storage_cond'] for b in batch]),
        }
