import os
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

class Stage1Dataset(Dataset):
    """第一阶段数据集：纯图像 + 标签（成熟度 / 新鲜度等级）

    数据来源：
      images_stage1/maturity_chashugan/{qingpi, erhong, dahong}  → 成熟度标签 0/1/2
      images_stage1/freshness_wenzhoumigan/{0d,7d,14d,21d,28d,35d,42d,49d} → 新鲜度标签 0-7

    两个子集合并为统一样本列表。成熟度子集的新鲜度标签置 -1，
    新鲜度子集的成熟度标签置 -1，训练时对 -1 标签跳过对应损失。
    """

    # 子文件夹名 → 标签索引
    _MATURITY_FOLDERS = {'qingpi': 0, 'erhong': 1, 'dahong': 2}
    _FRESHNESS_FOLDERS = {'0d': 0, '7d': 1, '14d': 2, '21d': 3,
                          '28d': 4, '35d': 5, '42d': 6, '49d': 7}

    def __init__(self, data_dir, transform=None, is_train=True, val_ratio=0.2, seed=42):
        """
        Args:
            data_dir:   数据根目录（包含 images_stage1/ 子目录）
            transform:  torchvision transforms，输入为 PIL Image
            is_train:   True → 训练集，False → 验证集
            val_ratio:  验证集比例
            seed:       随机种子，保证划分可复现
        """
        self.transform = transform

        maturity_root = os.path.join(data_dir, 'images_stage1', 'maturity_chashugan')
        freshness_root = os.path.join(data_dir, 'images_stage1', 'freshness_wenzhoumigan')

        samples = []  # list of (img_path, maturity_label, freshness_label)

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

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(samples))
        split = int(len(samples) * (1 - val_ratio))
        chosen = indices[:split] if is_train else indices[split:]
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
            'image': img,
            'maturity': torch.tensor(maturity, dtype=torch.long),
            'freshness': torch.tensor(freshness, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

# 时间点（天）顺序，与 Excel 行顺序一致（共 9 个时期）
_TIME_POINTS = [3, 6, 9, 12, 15, 20, 25, 35, 45]

# 图像文件夹只有 7 个时间点（无 6天、12天）
# 映射：时间点 → 最近可用图像文件夹名
_IMG_FOLDER_MAP = {
    3: '3天', 6: '9天',   # 6天 缺失，用 9天 代替
    9: '9天', 12: '15天', # 12天 缺失，用 15天 代替
    15: '15天', 20: '20天', 25: '25天', 35: '35天', 45: '45天',
}


def _load_excel_series(path, col_idx):
    """从 Excel 中读取指定列的 9 个时间点数值序列。

    支持两种格式：
      - 有标题行（第0行=表名，第1行=列名，第2行起=数据）：fubailv/meihuoxing/shizhonglv/shuifen
      - 无标题行（第0行=列名，第1行起=数据）：yingdu
    自动跳过非数值行，取最后连续的 9 个数值行。
    返回 shape=(9,) 的 float32 数组，顺序与 _TIME_POINTS 一致。
    """
    df = pd.read_excel(path, header=None)
    # 取数值列，强制转换，非数值变 NaN，然后丢弃 NaN 行
    series = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna()
    values = series.values
    assert len(values) == 9, f"{path} 期望9个数值，实际{len(values)}个"
    return values.astype(np.float32)


def _load_dianzibi(path, col_idx):
    """读取电子鼻 Excel（dianzibi.xlsx）中指定存储条件的 9×8 矩阵。

    文件结构：两个数据块（常温/冷藏），各占 9 行，中间有空行和标题行。
    col_idx=0 → 常温（行 3-11），col_idx=1 → 冷藏（行 15-23）。
    返回 shape=(9, 8) 的 float32 数组。
    """
    df = pd.read_excel(path, header=None)
    if col_idx == 0:
        block = df.iloc[3:12, 1:9]   # 常温：行索引 3-11，列索引 1-8
    else:
        block = df.iloc[15:24, 1:9]  # 冷藏：行索引 15-23，列索引 1-8
    return block.astype(float).values.astype(np.float32)  # (9, 8)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

class Stage2Dataset(Dataset):
    """第二阶段数据集：多模态时序数据（9个时期）

    数据来源：
      baoxian_stage2/changwen/  → 常温存储图像（storage_cond=0），7个时间点
      baoxian_stage2/lengcang/  → 冷藏存储图像（storage_cond=1），7个时间点
      baoxian_stage2/dianzibi.xlsx   → 电子鼻，shape (9, 8)
      baoxian_stage2/meihuoxing.xlsx → 酶活性，shape (9,)
      baoxian_stage2/shizhonglv.xlsx → 失重率，shape (9,)
      baoxian_stage2/shuifen.xlsx    → 水分，shape (9,)
      baoxian_stage2/fubailv.xlsx    → 腐败率，shape (9,)
      baoxian_stage2/yingdu.xlsx     → 硬度，shape (9,)

    每个样本对应一种存储条件（共 2 个），通过数据增强扩充样本数。
    et_seq 形状：[T, 1, 8]（每时间点 1 次电子鼻测量，8 个传感器通道）。
    """

    def __init__(self, data_dir, config, is_train=True, transform=None):
        """
        Args:
            data_dir:  数据根目录（包含 baoxian_stage2/ 子目录）
            config:    Config 对象
            is_train:  保留参数，与 Stage1Dataset 接口一致（Stage2 样本量少，不做划分）
            transform: 应用于每帧图像的 torchvision transforms（输入 PIL Image）
        """
        self.config = config
        self.transform = transform
        self.T = config.num_time_steps  # 9
        self.img_size = config.img_size

        stage2_root = os.path.join(data_dir, 'baoxian_stage2')

        # 预加载所有 Excel 数据（两种存储条件）
        dianzibi_path   = os.path.join(stage2_root, 'dianzibi.xlsx')
        meihuoxing_path = os.path.join(stage2_root, 'meihuoxing.xlsx')
        shizhonglv_path = os.path.join(stage2_root, 'shizhonglv.xlsx')
        shuifen_path    = os.path.join(stage2_root, 'shuifen.xlsx')
        fubailv_path    = os.path.join(stage2_root, 'fubailv.xlsx')
        yingdu_path     = os.path.join(stage2_root, 'yingdu.xlsx')

        # cond_idx: 0=常温, 1=冷藏
        self.et_data   = [_load_dianzibi(dianzibi_path, c) for c in range(2)]    # list of (9,8)
        self.enz_data  = [_load_excel_series(meihuoxing_path, c + 1) for c in range(2)]  # list of (9,)
        self.wl_data   = [_load_excel_series(shizhonglv_path, c + 1) for c in range(2)]
        self.moist_data= [_load_excel_series(shuifen_path, c + 1) for c in range(2)]
        self.fcr_data  = [_load_excel_series(fubailv_path, c + 1) for c in range(2)]
        self.yingdu_data=[_load_excel_series(yingdu_path, c + 1) for c in range(2)]

        # 图像路径：cond → list of (time_point, img_path)
        cond_folders = {0: 'changwen', 1: 'lengcang'}
        self.img_paths = {}  # cond → {folder_name: path}
        for cond, folder in cond_folders.items():
            folder_path = os.path.join(stage2_root, folder)
            imgs = {}
            for p in sorted(glob.glob(os.path.join(folder_path, '*.JPG')) +
                            glob.glob(os.path.join(folder_path, '*.jpg'))):
                name = os.path.splitext(os.path.basename(p))[0]  # e.g. "3天"
                imgs[name] = p
            self.img_paths[cond] = imgs

        # 每个存储条件作为一个样本（共 2 个）
        self.conditions = [0, 1]

    def __len__(self):
        return len(self.conditions)

    def _load_img_seq(self, cond):
        """加载指定存储条件的图像序列，返回 [T, 3, H, W] tensor。"""
        imgs = self.img_paths[cond]
        frames = []
        for t in _TIME_POINTS:
            folder_name = _IMG_FOLDER_MAP[t]
            img_path = imgs[folder_name]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                frame = self.transform(img)
            else:
                img = img.resize((self.img_size, self.img_size))
                frame = torch.from_numpy(
                    np.array(img, dtype=np.float32) / 255.0
                ).permute(2, 0, 1)
            frames.append(frame)
        return torch.stack(frames)  # [T, 3, H, W]

    def __getitem__(self, idx):
        cond = self.conditions[idx]

        img_seq  = self._load_img_seq(cond)                          # [T, 3, H, W]
        et_seq   = torch.from_numpy(self.et_data[cond])              # [T, 8]
        et_seq   = et_seq.unsqueeze(1)                               # [T, 1, 8]
        enz_seq  = torch.from_numpy(self.enz_data[cond])             # [T]
        wl_seq   = torch.from_numpy(self.wl_data[cond])              # [T]

        storage_cond = torch.tensor(cond, dtype=torch.long)

        # 目标值取最后一个时期（45天）
        fcr      = torch.tensor(self.fcr_data[cond][-1],   dtype=torch.float32)
        moisture = torch.tensor(self.moist_data[cond][-1], dtype=torch.float32)
        hardness = torch.tensor(self.yingdu_data[cond][-1],dtype=torch.float32)

        # 静态属性：Stage2 数据集无成熟度/新鲜度标注，置 -1
        maturity = torch.tensor(-1, dtype=torch.long)
        freshness = torch.tensor(-1.0, dtype=torch.float32)

        return {
            'img_seq':      img_seq,
            'et_seq':       et_seq,
            'enz_seq':      enz_seq,
            'wl_seq':       wl_seq,
            'maturity':     maturity,
            'freshness':    freshness,
            'storage_cond': storage_cond,
            'fcr':          fcr,
            'moisture':     moisture,
            'hardness':     hardness,
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'img_seq':      torch.stack([b['img_seq']      for b in batch]),
            'et_seq':       torch.stack([b['et_seq']       for b in batch]),
            'enz_seq':      torch.stack([b['enz_seq']      for b in batch]),
            'wl_seq':       torch.stack([b['wl_seq']       for b in batch]),
            'maturity':     torch.stack([b['maturity']     for b in batch]),
            'freshness':    torch.stack([b['freshness']    for b in batch]),
            'storage_cond': torch.stack([b['storage_cond'] for b in batch]),
            'fcr':          torch.stack([b['fcr']          for b in batch]),
            'moisture':     torch.stack([b['moisture']     for b in batch]),
            'hardness':     torch.stack([b['hardness']     for b in batch]),
        }
