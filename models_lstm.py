"""
对比实验模型：图像 + 电子舌（简单拼接 + LSTM）
- 输入：图像序列 + 电子舌序列，不涉及理化数据（酶活性、失重率）
- 微观编码：电子舌直接 Flatten 后与图像特征拼接
- 宏观时序：双向 LSTM
- 输出：腐败率、水分、硬度
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# 图像编码器（ResNet50，与原模型相同）
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features   = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.projection(x)


# ---------------------------------------------------------------------------
# 电子舌编码器：Flatten + Linear（无序列建模）
# ---------------------------------------------------------------------------

class FlatETEncoder(nn.Module):
    """将电子舌 [B, seq_len, channels] 直接展平后线性映射
    seq_len=1, channels=8 → in_dim=8
    """

    def __init__(self, seq_len=1, channels=8, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, seq_len, channels]
        return self.net(x)


# ---------------------------------------------------------------------------
# 单时间步拼接融合：图像 + 电子舌（不含理化数据）
# ---------------------------------------------------------------------------

class TimeStepConcat(nn.Module):
    """将图像特征与电子舌特征直接拼接后线性映射"""

    def __init__(self, img_dim=512, et_dim=128, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(img_dim + et_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, img_feat, et_feat):
        return self.proj(torch.cat([img_feat, et_feat], dim=1))


# ---------------------------------------------------------------------------
# 宏观时序建模：双向 LSTM
# ---------------------------------------------------------------------------

class TemporalLSTM(nn.Module):
    """双向 LSTM 时序编码器
    输入: [B, T, input_dim]
    输出: [B, T, hidden_dim]
    """

    def __init__(self, input_dim=256, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,   # 双向拼接后恢复 hidden_dim
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq):
        # seq: [B, T, input_dim]
        out, _ = self.lstm(seq)   # [B, T, hidden_dim]
        return self.norm(out)


# ---------------------------------------------------------------------------
# 静态条件编码器（与原模型相同）
# ---------------------------------------------------------------------------

class StaticConditionEncoder(nn.Module):
    def __init__(self, maturity_classes=3, storage_cond_dim=2, out_dim=128):
        super().__init__()
        self.maturity_embed = nn.Embedding(maturity_classes, 16)
        self.storage_embed  = nn.Embedding(storage_cond_dim, 8)
        self.mlp = nn.Sequential(
            nn.Linear(16 + 1 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, maturity, freshness, storage_cond):
        maturity_valid = maturity.clamp(min=0)
        m_emb = self.maturity_embed(maturity_valid)
        m_emb = m_emb * (maturity >= 0).float().unsqueeze(-1)
        f     = freshness.clamp(min=0).unsqueeze(-1)
        s_emb = self.storage_embed(storage_cond)
        return self.mlp(torch.cat([m_emb, f, s_emb], dim=1))


# ---------------------------------------------------------------------------
# 多任务输出头（与原模型相同）
# ---------------------------------------------------------------------------

class MultiTaskOutputHead(nn.Module):
    def __init__(self, input_dim=640, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fcr_head      = nn.Linear(hidden_dim, 1)
        self.moisture_head = nn.Linear(hidden_dim, 1)
        self.hardness_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        s = self.shared(x)
        return self.fcr_head(s), self.moisture_head(s), self.hardness_head(s)


# ---------------------------------------------------------------------------
# Stage1 模型（与原模型相同，用于预训练图像编码器）
# ---------------------------------------------------------------------------

class Stage1Model(nn.Module):
    def __init__(self, img_encoder, num_maturity=3, num_freshness=8):
        super().__init__()
        self.img_encoder    = img_encoder
        self.maturity_head  = nn.Linear(512, num_maturity)
        self.freshness_head = nn.Linear(512, num_freshness)

    def forward(self, img):
        feat = self.img_encoder(img)
        return self.maturity_head(feat), self.freshness_head(feat), feat


# ---------------------------------------------------------------------------
# Stage2 对比模型：图像 + 电子舌简单拼接 + 双向 LSTM（无理化数据）
# ---------------------------------------------------------------------------

class Stage2LSTMModel(nn.Module):
    """对比实验 Stage2 模型
    输入：图像序列 + 电子舌序列（不含酶活性、失重率等理化数据）
    微观编码：Flatten 直接拼接
    宏观时序：双向 LSTM
    """

    def __init__(self, stage1_encoder, config):
        super().__init__()
        self.img_encoder = stage1_encoder

        # 电子舌：直接展平拼接
        self.et_encoder = FlatETEncoder(
            seq_len=config.et_seq_len,
            channels=config.et_channels,
            out_dim=config.et_feat_dim
        )

        # 时间步拼接融合（仅图像 + 电子舌，无理化数据）
        self.step_concat = TimeStepConcat(
            img_dim=config.img_feat_dim,
            et_dim=config.et_feat_dim,
            out_dim=config.step_fusion_dim
        )

        # 宏观：双向 LSTM
        self.temporal_lstm = TemporalLSTM(
            input_dim=config.step_fusion_dim,
            hidden_dim=config.temp_hidden_dim,
            num_layers=2,
            dropout=config.dropout
        )

        # 静态条件编码
        self.static_encoder = StaticConditionEncoder(
            maturity_classes=config.maturity_classes,
            storage_cond_dim=2,
            out_dim=config.static_feat_dim
        )

        # 多任务输出头
        self.output_head = MultiTaskOutputHead(
            input_dim=config.temp_hidden_dim + config.static_feat_dim,
            hidden_dim=256
        )

    def forward(self, img_seq, et_seq, maturity, freshness, storage_cond):
        """
        img_seq:      [B, T, 3, H, W]
        et_seq:       [B, T, seq_len, channels]
        maturity:     [B]
        freshness:    [B]
        storage_cond: [B]
        注意：不接收 enz_seq 和 wl_seq
        """
        B, T = img_seq.shape[:2]
        step_feats = []

        for t in range(T):
            img = img_seq[:, t]   # [B, 3, H, W]
            et  = et_seq[:, t]    # [B, seq_len, channels]

            img_feat  = self.img_encoder(img)  # [B, 512]
            et_feat   = self.et_encoder(et)    # [B, 128]

            step_feat = self.step_concat(img_feat, et_feat)  # [B, 256]
            step_feats.append(step_feat)

        seq      = torch.stack(step_feats, dim=1)     # [B, T, 256]
        lstm_out = self.temporal_lstm(seq)             # [B, T, hidden_dim]
        last_out = lstm_out[:, -1, :]                  # [B, hidden_dim]

        static_feat = self.static_encoder(maturity, freshness, storage_cond)  # [B, 128]
        combined    = torch.cat([last_out, static_feat], dim=1)               # [B, 640]

        fcr, moisture, hardness = self.output_head(combined)
        return fcr, moisture, hardness
