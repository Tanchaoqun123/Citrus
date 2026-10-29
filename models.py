import torch
import torch.nn as nn
import torchvision.models as models

# 尝试导入真实 mamba_ssm，失败则使用简化版（LSTM）
try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
    print("Using real Mamba from mamba_ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    print("Mamba not available, using simplified alternative (LSTM based)")


    class SimpleMamba(nn.Module):
        """简化版Mamba (用于无CUDA环境)"""

        def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
            super().__init__()
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(d_model * 2, d_model)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.proj(out)
            return out


    MambaClass = SimpleMamba
else:
    MambaClass = Mamba


class ImageEncoder(nn.Module):
    """图像编码器（ResNet50）"""

    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # 去掉分类头，保留特征提取部分
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
        x = self.projection(x)
        return x


class MambaETEncoder(nn.Module):
    """电子舌序列编码器（微观Mamba：处理每个时间步内的电子舌信号）"""

    def __init__(self, input_dim=6, d_model=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba = MambaClass(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(d_model, 128)

    def forward(self, x):
        # x: [B, T, C]
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.mamba(x)  # [B, T, d_model]
        x = x.permute(0, 2, 1)  # [B, d_model, T]
        x = self.pool(x).squeeze(-1)  # [B, d_model]
        x = self.out_proj(x)  # [B, 128]
        return x


class ScalarEncoder(nn.Module):
    """标量值编码器（酶活性、失重率等）"""

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


class TimeStepFusion(nn.Module):
    """单个时间步内多模态融合"""

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


class TemporalMamba(nn.Module):
    """宏观时序建模：使用Mamba处理跨时间步的特征序列（双Mamba核心）"""

    def __init__(self, input_dim=256, hidden_dim=512, d_state=16, d_conv=4, expand=2, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba_layers = nn.ModuleList([
            MambaClass(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand_factor=expand)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, seq):
        # seq: [B, T, input_dim]
        x = self.input_proj(seq)  # [B, T, hidden_dim]
        for mamba in self.mamba_layers:
            x = mamba(x)  # [B, T, hidden_dim]
        x = self.norm(x)
        x = self.out_proj(x)  # [B, T, hidden_dim]
        return x


class StaticConditionEncoder(nn.Module):
    """静态属性编码（成熟度、新鲜度、存储条件）"""

    def __init__(self, maturity_classes=3, storage_cond_dim=2, out_dim=128):
        super().__init__()
        self.maturity_embed = nn.Embedding(maturity_classes, 16)
        self.storage_embed = nn.Embedding(storage_cond_dim, 8)
        total_dim = 16 + 1 + 8  # 新鲜度是1维连续值
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, maturity, freshness, storage_cond):
        # maturity: [B], freshness: [B] (normalized 0~1), storage_cond: [B]
        # maturity=-1 means unknown (Stage2 samples); clamp to 0 then zero the embedding
        maturity_valid = maturity.clamp(min=0)
        m_emb = self.maturity_embed(maturity_valid)  # [B,16]
        m_emb = m_emb * (maturity >= 0).float().unsqueeze(-1)  # zero out unknown
        f = freshness.clamp(min=0).unsqueeze(-1)  # [B,1], -1 → 0
        s_emb = self.storage_embed(storage_cond)  # [B,8]
        combined = torch.cat([m_emb, f, s_emb], dim=1)
        return self.mlp(combined)


class MultiTaskOutputHead(nn.Module):
    """多任务输出头（腐败率、水分、硬度）"""

    def __init__(self, input_dim=512 + 128, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fcr_head = nn.Linear(hidden_dim, 1)
        self.moisture_head = nn.Linear(hidden_dim, 1)
        self.hardness_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.fcr_head(shared), self.moisture_head(shared), self.hardness_head(shared)


class Stage1Model(nn.Module):
    """第一阶段：纯图像多任务模型"""

    def __init__(self, img_encoder, num_maturity=3, num_freshness=3):
        super().__init__()
        self.img_encoder = img_encoder
        self.maturity_head = nn.Linear(512, num_maturity)
        self.freshness_head = nn.Linear(512, num_freshness)

    def forward(self, img):
        feat = self.img_encoder(img)
        maturity = self.maturity_head(feat)
        freshness = self.freshness_head(feat)
        return maturity, freshness, feat


class Stage2Model(nn.Module):
    """第二阶段：双Mamba多模态时序预测模型（微观Mamba + 宏观Mamba）"""

    def __init__(self, stage1_encoder, config):
        super().__init__()
        # 加载第一阶段图像编码器（可部分微调）
        self.img_encoder = stage1_encoder

        # 微观Mamba：电子舌编码器
        self.et_encoder = MambaETEncoder(
            input_dim=config.et_channels,
            d_model=128,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand
        )
        # 标量编码器
        self.enc_enz = ScalarEncoder(in_dim=1, out_dim=config.scalar_feat_dim)
        self.enc_wl = ScalarEncoder(in_dim=1, out_dim=config.scalar_feat_dim)
        # 时间步融合
        self.step_fusion = TimeStepFusion(
            img_dim=config.img_feat_dim,
            et_dim=config.et_feat_dim,
            scalar_dim=config.scalar_feat_dim,
            out_dim=config.step_fusion_dim
        )
        # 宏观Mamba：跨时间步时序建模
        self.temporal_mamba = TemporalMamba(
            input_dim=config.step_fusion_dim,
            hidden_dim=config.temp_hidden_dim,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            num_layers=config.temporal_mamba_layers
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
        img_seq: [B, T, 3, H, W]
        et_seq:  [B, T, T_et, C_et]
        enz_seq: [B, T]
        wl_seq:  [B, T]
        maturity: [B]
        freshness: [B]
        storage_cond: [B]
        """
        B, T = img_seq.shape[:2]
        step_feats = []
        for t in range(T):
            img = img_seq[:, t]  # [B,3,H,W]
            et = et_seq[:, t]  # [B, T_et, C_et]
            enz = enz_seq[:, t].unsqueeze(-1)  # [B,1]
            wl = wl_seq[:, t].unsqueeze(-1)  # [B,1]

            img_feat = self.img_encoder(img)  # [B,512]
            et_feat = self.et_encoder(et)  # [B,128]
            enz_feat = self.enc_enz(enz)  # [B,32]
            wl_feat = self.enc_wl(wl)  # [B,32]

            step_feat = self.step_fusion(img_feat, et_feat, enz_feat, wl_feat)  # [B,256]
            step_feats.append(step_feat)

        seq = torch.stack(step_feats, dim=1)  # [B, T, 256]
        temp_out = self.temporal_mamba(seq)  # [B, T, temp_hidden_dim]
        last_temp = temp_out[:, -1, :]  # [B, temp_hidden_dim]

        static_feat = self.static_encoder(maturity, freshness, storage_cond)  # [B,128]
        combined = torch.cat([last_temp, static_feat], dim=1)  # [B, temp_hidden_dim+128]
        fcr, moisture, hardness = self.output_head(combined)
        return fcr, moisture, hardness