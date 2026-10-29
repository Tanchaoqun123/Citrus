"""
对比实验模型：图像 + 电子鼻 + 理化（Transformer时序）
微观编码：MLP
宏观时序：Transformer Encoder
"""

import torch
import torch.nn as nn
import torchvision.models as models
import math


# ---------------------------------------------------------------------------
# 图像编码器（与原模型相同结构，独立定义）
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """ResNet50 图像编码器"""

    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # [B, 2048, 7, 7]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
# 微观编码：MLP（替换原模型的 MambaETEncoder）
# ---------------------------------------------------------------------------

class MLPETEncoder(nn.Module):
    """电子鼻微观编码器：MLP
    输入: [B, seq_len, channels]  (seq_len=1, channels=8)
    输出: [B, out_dim]
    """

    def __init__(self, seq_len=1, channels=8, hidden_dim=64, out_dim=128):
        super().__init__()
        in_dim = seq_len * channels
        self.net = nn.Sequential(
            nn.Flatten(),                          # [B, seq_len*channels]
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: [B, seq_len, channels]
        return self.net(x)


# ---------------------------------------------------------------------------
# 标量编码器（与原模型相同）
# ---------------------------------------------------------------------------

class ScalarEncoder(nn.Module):
    """标量值编码器（酶活性、失重率）"""

    def __init__(self, in_dim=1, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 单时间步多模态融合（与原模型相同）
# ---------------------------------------------------------------------------

class TimeStepFusion(nn.Module):
    """单个时间步内多模态特征融合"""

    def __init__(self, img_dim=512, et_dim=128, scalar_dim=32, out_dim=256):
        super().__init__()
        total_dim = img_dim + et_dim + scalar_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim),
            nn.ReLU()
        )

    def forward(self, img_feat, et_feat, enz_feat, wl_feat):
        combined = torch.cat([img_feat, et_feat, enz_feat, wl_feat], dim=1)
        return self.fusion(combined)


# ---------------------------------------------------------------------------
# 宏观时序编码：Transformer Encoder（替换原模型的 TemporalMamba）
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """宏观时序建模：Transformer Encoder
    输入: [B, T, input_dim]
    输出: [B, T, hidden_dim]
    """

    def __init__(self, input_dim=256, hidden_dim=512, nhead=8,
                 num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq):
        # seq: [B, T, input_dim]
        x = self.input_proj(seq)       # [B, T, hidden_dim]
        x = self.pos_enc(x)            # 加位置编码
        x = self.transformer(x)        # [B, T, hidden_dim]
        return self.norm(x)            # [B, T, hidden_dim]


# ---------------------------------------------------------------------------
# 静态条件编码器（与原模型相同）
# ---------------------------------------------------------------------------

class StaticConditionEncoder(nn.Module):
    """静态属性编码（成熟度、新鲜度、存储条件）"""

    def __init__(self, maturity_classes=3, storage_cond_dim=2, out_dim=128):
        super().__init__()
        self.maturity_embed = nn.Embedding(maturity_classes, 16)
        self.storage_embed = nn.Embedding(storage_cond_dim, 8)
        total_dim = 16 + 1 + 8
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, maturity, freshness, storage_cond):
        maturity_valid = maturity.clamp(min=0)
        m_emb = self.maturity_embed(maturity_valid)
        m_emb = m_emb * (maturity >= 0).float().unsqueeze(-1)
        f = freshness.clamp(min=0).unsqueeze(-1)
        s_emb = self.storage_embed(storage_cond)
        combined = torch.cat([m_emb, f, s_emb], dim=1)
        return self.mlp(combined)


# ---------------------------------------------------------------------------
# 多任务输出头（与原模型相同）
# ---------------------------------------------------------------------------

class MultiTaskOutputHead(nn.Module):
    """多任务输出头（腐败率、水分、硬度）"""

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
        shared = self.shared(x)
        return self.fcr_head(shared), self.moisture_head(shared), self.hardness_head(shared)


# ---------------------------------------------------------------------------
# Stage1 模型（与原模型相同，用于预训练图像编码器）
# ---------------------------------------------------------------------------

class Stage1Model(nn.Module):
    """第一阶段：纯图像多任务分类模型"""

    def __init__(self, img_encoder, num_maturity=3, num_freshness=8):
        super().__init__()
        self.img_encoder   = img_encoder
        self.maturity_head = nn.Linear(512, num_maturity)
        self.freshness_head = nn.Linear(512, num_freshness)

    def forward(self, img):
        feat = self.img_encoder(img)
        return self.maturity_head(feat), self.freshness_head(feat), feat


# ---------------------------------------------------------------------------
# Stage2 对比模型：MLP微观 + Transformer宏观
# ---------------------------------------------------------------------------

class Stage2TransformerModel(nn.Module):
    """对比实验 Stage2 模型
    微观编码：MLP（电子鼻）
    宏观时序：Transformer Encoder
    其余结构与原 Stage2Model 保持一致
    """

    def __init__(self, stage1_encoder, config):
        super().__init__()
        self.img_encoder = stage1_encoder

        # 微观：MLP 电子鼻编码器
        self.et_encoder = MLPETEncoder(
            seq_len=config.et_seq_len,
            channels=config.et_channels,
            hidden_dim=64,
            out_dim=config.et_feat_dim
        )

        # 标量编码器
        self.enc_enz = ScalarEncoder(in_dim=1, out_dim=config.scalar_feat_dim)
        self.enc_wl  = ScalarEncoder(in_dim=1, out_dim=config.scalar_feat_dim)

        # 时间步融合
        self.step_fusion = TimeStepFusion(
            img_dim=config.img_feat_dim,
            et_dim=config.et_feat_dim,
            scalar_dim=config.scalar_feat_dim,
            out_dim=config.step_fusion_dim
        )

        # 宏观：Transformer 时序编码器
        # nhead 需能整除 hidden_dim，取 8
        self.temporal_transformer = TemporalTransformer(
            input_dim=config.step_fusion_dim,
            hidden_dim=config.temp_hidden_dim,
            nhead=8,
            num_layers=2,
            dim_feedforward=config.temp_hidden_dim * 2,
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

    def forward(self, img_seq, et_seq, enz_seq, wl_seq, maturity, freshness, storage_cond):
        """
        img_seq:      [B, T, 3, H, W]
        et_seq:       [B, T, seq_len, channels]
        enz_seq:      [B, T]
        wl_seq:       [B, T]
        maturity:     [B]
        freshness:    [B]
        storage_cond: [B]
        """
        B, T = img_seq.shape[:2]
        step_feats = []

        for t in range(T):
            img  = img_seq[:, t]              # [B, 3, H, W]
            et   = et_seq[:, t]               # [B, seq_len, channels]
            enz  = enz_seq[:, t].unsqueeze(-1) # [B, 1]
            wl   = wl_seq[:, t].unsqueeze(-1)  # [B, 1]

            img_feat = self.img_encoder(img)   # [B, 512]
            et_feat  = self.et_encoder(et)     # [B, 128]
            enz_feat = self.enc_enz(enz)       # [B, 32]
            wl_feat  = self.enc_wl(wl)         # [B, 32]

            step_feat = self.step_fusion(img_feat, et_feat, enz_feat, wl_feat)  # [B, 256]
            step_feats.append(step_feat)

        seq      = torch.stack(step_feats, dim=1)          # [B, T, 256]
        temp_out = self.temporal_transformer(seq)           # [B, T, hidden_dim]
        last_out = temp_out[:, -1, :]                       # [B, hidden_dim]

        static_feat = self.static_encoder(maturity, freshness, storage_cond)  # [B, 128]
        combined    = torch.cat([last_out, static_feat], dim=1)               # [B, hidden_dim+128]

        fcr, moisture, hardness = self.output_head(combined)
        return fcr, moisture, hardness
