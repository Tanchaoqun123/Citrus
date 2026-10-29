import os
import torch


class Config:
    # 路径配置（请根据实际情况修改）
    data_root = './data'
    stage1_save_dir = './checkpoints/stage1'
    stage2_save_dir = './checkpoints/stage2'
    log_dir = './logs'

    # 第一阶段配置（纯图像多任务）
    stage1_batch_size = 32
    stage1_epochs = 50
    stage1_lr = 1e-4
    stage1_num_workers = 2

    # 第二阶段配置（多模态时序预测）
    stage2_batch_size = 2
    stage2_epochs = 100
    stage2_lr = 5e-5
    stage2_num_workers = 0

    # 数据参数py
    num_time_steps = 9  # 9个时期
    img_size = 224
    et_seq_len = 1   # 电子鼻每时间点测量次数（dianzibi.xlsx 每时间点1条记录）
    et_channels = 8  # 电子鼻传感器通道数（dianzibi.xlsx 实际8列）
    maturity_classes = 3
    freshness_classes = 8

    # 模型维度参数
    img_feat_dim = 512
    et_feat_dim = 128
    scalar_feat_dim = 32
    step_fusion_dim = 256
    temp_hidden_dim = 512  # 时序Mamba的隐藏维度
    static_feat_dim = 128
    # Mamba
    mamba_d_state = 16
    mamba_d_conv = 4
    mamba_expand = 2
    temporal_mamba_layers = 2  # 宏观Mamba堆叠层数

    dropout = 0.2

    # 损失权重
    loss_weights_stage1 = {'maturity': 1.0, 'freshness': 1.0}
    loss_weights_stage2 = {'fcr': 1.0, 'moisture': 0.8, 'hardness': 0.6}

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def create_dirs():
        for d in [Config.stage1_save_dir, Config.stage2_save_dir, Config.log_dir]:
            os.makedirs(d, exist_ok=True)