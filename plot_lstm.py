"""
对比实验结果可视化：图像 + 电子舌（简单拼接 + LSTM）
计算 MAE、RMSE，按每个样本分别绘制 GT vs Pred 曲线
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

COMP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(COMP_DIR, 'logs_lstm', 'stage2_log.txt')
OUT_IMG  = os.path.join(COMP_DIR, 'logs_lstm', 'stage2_prediction_curves.png')
OUT_TXT  = os.path.join(COMP_DIR, 'logs_lstm', 'stage2_accuracy.txt')


# ---------------------------------------------------------------------------
# 日志解析
# ---------------------------------------------------------------------------

def parse_log(log_path):
    epochs = []
    current_epoch = None
    current_samples = []

    epoch_re  = re.compile(r'Epoch\s+(\d+):.*Val Loss=')
    sample_re = re.compile(
        r'\[\s*\d+\]\s*\|\s*(-?[\d.]+)\s*/\s*(-?[\d.]+)'
        r'\s*\|\s*(-?[\d.]+)\s*/\s*(-?[\d.]+)'
        r'\s*\|\s*(-?[\d.]+)\s*/\s*(-?[\d.]+)'
    )

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            em = epoch_re.search(line)
            if em:
                if current_epoch is not None and current_samples:
                    epochs.append({'epoch': current_epoch, 'samples': current_samples})
                current_epoch   = int(em.group(1))
                current_samples = []
                continue
            sm = sample_re.search(line)
            if sm and current_epoch is not None:
                current_samples.append({
                    'fcr_gt':     float(sm.group(1)),
                    'fcr_pred':   float(sm.group(2)),
                    'moist_gt':   float(sm.group(3)),
                    'moist_pred': float(sm.group(4)),
                    'hard_gt':    float(sm.group(5)),
                    'hard_pred':  float(sm.group(6)),
                })

    if current_epoch is not None and current_samples:
        epochs.append({'epoch': current_epoch, 'samples': current_samples})

    return epochs


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def calc_mae(gt, pred):
    gt, pred = np.array(gt, dtype=float), np.array(pred, dtype=float)
    return float(np.mean(np.abs(gt - pred)))


def calc_rmse(gt, pred):
    gt, pred = np.array(gt, dtype=float), np.array(pred, dtype=float)
    return float(np.sqrt(np.mean((gt - pred) ** 2)))


def epoch_metrics(epochs_data, key_gt, key_pred):
    """按 epoch 提取每个样本的 Pred 序列（GT 固定不变），以及 MAE、RMSE。
    返回:
        gt_values     : list，每个样本的 GT 固定值
        pred_by_sample: list of list，pred_by_sample[i] 是第 i 个样本在所有 epoch 的预测值
        maes          : list，每个 epoch 的 MAE
        rmses         : list，每个 epoch 的 RMSE
    """
    n_samples = len(epochs_data[0]['samples'])
    gt_values = [epochs_data[0]['samples'][i][key_gt] for i in range(n_samples)]

    pred_by_sample = [[] for _ in range(n_samples)]
    maes, rmses = [], []

    for e in epochs_data:
        gt_ep   = [s[key_gt]   for s in e['samples']]
        pred_ep = [s[key_pred] for s in e['samples']]
        maes.append(calc_mae(gt_ep, pred_ep))
        rmses.append(calc_rmse(gt_ep, pred_ep))
        for i in range(n_samples):
            pred_by_sample[i].append(pred_ep[i])

    return gt_values, pred_by_sample, maes, rmses


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LOG_PATH):
        print(f"Log file not found: {LOG_PATH}")
        return

    epochs_data = parse_log(LOG_PATH)
    if not epochs_data:
        print("No valid epoch data found in log.")
        return

    epoch_nums = [e['epoch'] for e in epochs_data]
    n_samples  = len(epochs_data[0]['samples'])

    fcr_gt,   fcr_pred_by_s,   fcr_mae,   fcr_rmse   = epoch_metrics(epochs_data, 'fcr_gt',   'fcr_pred')
    moist_gt, moist_pred_by_s, moist_mae, moist_rmse = epoch_metrics(epochs_data, 'moist_gt', 'moist_pred')
    hard_gt,  hard_pred_by_s,  hard_mae,  hard_rmse  = epoch_metrics(epochs_data, 'hard_gt',  'hard_pred')

    final = {
        'FCR':      {'mae': np.mean(fcr_mae),   'rmse': np.mean(fcr_rmse)},
        'Moisture': {'mae': np.mean(moist_mae), 'rmse': np.mean(moist_rmse)},
        'Hardness': {'mae': np.mean(hard_mae),  'rmse': np.mean(hard_rmse)},
    }

    # 打印并写入文件
    lines = [
        "=== [LSTM] Stage2 Final Prediction Accuracy ===",
        "",
        "MAE (Mean Absolute Error)   MAE  = mean(|GT - Pred|)",
        f"  FCR      : {final['FCR']['mae']:.4f}",
        f"  Moisture : {final['Moisture']['mae']:.4f}",
        f"  Hardness : {final['Hardness']['mae']:.4f}",
        "",
        "RMSE (Root Mean Square Error)   RMSE = sqrt(mean((GT - Pred)²))",
        f"  FCR      : {final['FCR']['rmse']:.4f}",
        f"  Moisture : {final['Moisture']['rmse']:.4f}",
        f"  Hardness : {final['Hardness']['rmse']:.4f}",
    ]
    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\nAccuracy report saved to: {OUT_TXT}")

    # 颜色方案
    sample_colors = ['tomato', 'steelblue', 'seagreen', 'darkorange', 'mediumpurple']
    gt_colors     = ['firebrick', 'navy', 'darkgreen', 'saddlebrown', 'indigo']

    # 绘图：3 行 × 3 列
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('[LSTM] Stage2 Prediction vs Ground Truth (per sample)', fontsize=14)

    metrics = [
        ('FCR',      fcr_gt,   fcr_pred_by_s,   fcr_mae,   fcr_rmse,   final['FCR']),
        ('Moisture', moist_gt, moist_pred_by_s, moist_mae, moist_rmse, final['Moisture']),
        ('Hardness', hard_gt,  hard_pred_by_s,  hard_mae,  hard_rmse,  final['Hardness']),
    ]

    for row, (name, gt_vals, pred_by_s, mae_vals, rmse_vals, fin) in enumerate(metrics):
        # 左：每个样本的 GT 水平线 + Pred 曲线
        ax = axes[row][0]
        for i in range(n_samples):
            color = sample_colors[i % len(sample_colors)]
            gc    = gt_colors[i % len(gt_colors)]
            ax.axhline(y=gt_vals[i], color=gc, linewidth=1.5,
                       linestyle='--', label=f'Sample{i+1} GT={gt_vals[i]:.4f}')
            ax.plot(epoch_nums, pred_by_s[i], color=color, linewidth=2,
                    label=f'Sample{i+1} Pred')
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} — GT vs Pred (per sample)', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

        # 中：MAE 随 epoch 变化
        ax = axes[row][1]
        ax.plot(epoch_nums, mae_vals,
                label=f'MAE (avg {fin["mae"]:.4f})', color='mediumpurple', linewidth=2)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(f'{name} — MAE over Epochs', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 右：RMSE 随 epoch 变化
        ax = axes[row][2]
        ax.plot(epoch_nums, rmse_vals,
                label=f'RMSE (avg {fin["rmse"]:.4f})', color='darkorange', linewidth=2)
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(f'{name} — RMSE over Epochs', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('Epoch', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=150)
    print(f"Plot saved to: {OUT_IMG}")


if __name__ == '__main__':
    main()
