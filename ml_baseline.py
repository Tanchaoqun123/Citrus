"""
传统机器学习建模：针对保鲜 Stage2 数据集
策略：将时间点展开为样本（2条序列 × 9时间点 = 18个样本）
特征：时间步 + 存储条件 + 电子鼻(8维) + 酶活性 + 失重率 = 12维
目标：腐败率(FCR)、水分(Moisture)、硬度(Hardness)
评估：Leave-One-Out 交叉验证
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 将原项目根目录加入路径
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_data(data_root):
    """加载 baoxian_stage2 下的所有 Excel 数据，展开为样本矩阵。

    返回:
        X : ndarray, shape (18, 12)  特征矩阵
        Y : ndarray, shape (18, 3)   目标矩阵 [fcr, moisture, hardness]
        feature_names : list
        target_names  : list
    """
    base = os.path.join(data_root, 'baoxian_stage2')

    # 时间点（天）
    time_points = [3, 6, 9, 12, 15, 20, 25, 35, 45]
    T = len(time_points)

    # ---------- 读取电子鼻数据 ----------
    def load_dianzibi(path, cond_idx):
        df = pd.read_excel(path, header=None)
        if cond_idx == 0:
            block = df.iloc[3:12, 1:9]   # 常温
        else:
            block = df.iloc[15:24, 1:9]  # 冷藏
        return block.astype(float).values  # (9, 8)

    # ---------- 读取标量序列 ----------
    def load_series(path, col_idx):
        df = pd.read_excel(path, header=None)
        series = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna()
        vals = series.values
        assert len(vals) == 9, f"{path} 期望9个值，实际{len(vals)}个"
        return vals.astype(float)

    et_data    = [load_dianzibi(os.path.join(base, 'dianzibi.xlsx'), c)   for c in range(2)]
    enz_data   = [load_series(os.path.join(base, 'meihuoxing.xlsx'), c+1) for c in range(2)]
    wl_data    = [load_series(os.path.join(base, 'shizhonglv.xlsx'), c+1) for c in range(2)]
    moist_data = [load_series(os.path.join(base, 'shuifen.xlsx'),    c+1) for c in range(2)]
    hard_data  = [load_series(os.path.join(base, 'yingdu.xlsx'),     c+1) for c in range(2)]
    fcr_data   = [load_series(os.path.join(base, 'fubailv.xlsx'),    c+1) for c in range(2)]

    rows_X, rows_Y = [], []

    for cond in range(2):           # 0=常温, 1=冷藏
        for t in range(T):
            t_norm = t / (T - 1)    # 归一化时间步 [0, 1]
            et_row = et_data[cond][t]           # (8,)
            enz    = enz_data[cond][t]
            wl     = wl_data[cond][t]

            feat = np.concatenate([
                [t_norm],           # 1 维：归一化时间
                [cond],             # 1 维：存储条件
                et_row,             # 8 维：电子鼻
                [enz],              # 1 维：酶活性
                [wl],               # 1 维：失重率
            ])                      # 共 12 维
            rows_X.append(feat)

            target = np.array([
                fcr_data[cond][t],
                moist_data[cond][t],
                hard_data[cond][t],
            ])
            rows_Y.append(target)

    X = np.array(rows_X)   # (18, 12)
    Y = np.array(rows_Y)   # (18, 3)

    feature_names = (
        ['time_norm', 'storage_cond'] +
        [f'et_{i+1}' for i in range(8)] +
        ['enzyme', 'weight_loss']
    )
    target_names = ['FCR', 'Moisture', 'Hardness']

    return X, Y, feature_names, target_names


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------

def get_models():
    return {
        'Ridge':
            Pipeline([('scaler', StandardScaler()),
                      ('model', Ridge(alpha=1.0))]),

        'Lasso':
            Pipeline([('scaler', StandardScaler()),
                      ('model', Lasso(alpha=0.01, max_iter=5000))]),

        'SVR_RBF':
            Pipeline([('scaler', StandardScaler()),
                      ('model', SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.01))]),

        'SVR_Linear':
            Pipeline([('scaler', StandardScaler()),
                      ('model', SVR(kernel='linear', C=1.0, epsilon=0.01))]),

        'RandomForest':
            Pipeline([('scaler', StandardScaler()),
                      ('model', RandomForestRegressor(
                          n_estimators=50, max_depth=4, random_state=42))]),

        'GradientBoosting':
            Pipeline([('scaler', StandardScaler()),
                      ('model', GradientBoostingRegressor(
                          n_estimators=50, max_depth=3, learning_rate=0.1,
                          random_state=42))]),

        'GaussianProcess':
            Pipeline([('scaler', StandardScaler()),
                      ('model', GaussianProcessRegressor(
                          kernel=ConstantKernel(1.0) * Matern(nu=1.5),
                          alpha=1e-2, normalize_y=True, random_state=42))]),
    }


# ---------------------------------------------------------------------------
# 评估：Leave-One-Out 交叉验证
# ---------------------------------------------------------------------------

def evaluate_loo(X, Y, target_names):
    """对每个目标变量分别用 LOO-CV 评估所有模型。"""
    loo = LeaveOneOut()
    models = get_models()
    results = {}   # results[model_name][target] = {'mae': ..., 'rmse': ...}

    for model_name, pipeline in models.items():
        results[model_name] = {}
        for j, tname in enumerate(target_names):
            y = Y[:, j]
            preds = []
            for train_idx, test_idx in loo.split(X):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                # 克隆 pipeline 避免状态污染
                from sklearn.base import clone
                pipe = clone(pipeline)
                pipe.fit(X_tr, y_tr)
                preds.append(pipe.predict(X_te)[0])

            preds = np.array(preds)
            mae  = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            results[model_name][tname] = {
                'mae': mae, 'rmse': rmse,
                'gt': y, 'pred': preds
            }

    return results


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------

def print_results(results, target_names, out_txt):
    lines = ["=== Traditional ML: LOO-CV Results ===", ""]

    # 汇总表
    header = f"{'Model':<20}" + "".join(
        f"  {t}_MAE   {t}_RMSE" for t in target_names
    )
    lines.append(header)
    lines.append("-" * len(header))

    best = {t: {'mae': float('inf'), 'model': ''} for t in target_names}

    for model_name, res in results.items():
        row = f"{model_name:<20}"
        for t in target_names:
            mae  = res[t]['mae']
            rmse = res[t]['rmse']
            row += f"  {mae:.4f}   {rmse:.4f}  "
            if mae < best[t]['mae']:
                best[t]['mae']   = mae
                best[t]['model'] = model_name
        lines.append(row)

    lines.append("")
    lines.append("=== Best Model per Target (by MAE) ===")
    for t in target_names:
        lines.append(f"  {t:<12}: {best[t]['model']}  MAE={best[t]['mae']:.4f}")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\nReport saved to: {out_txt}")

    return best


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_results(results, target_names, out_img):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    n_models  = len(results)
    n_targets = len(target_names)
    model_names = list(results.keys())

    fig, axes = plt.subplots(n_targets, 2, figsize=(14, 4 * n_targets))
    fig.suptitle('Traditional ML: LOO-CV Prediction Results', fontsize=13)

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for j, tname in enumerate(target_names):
        ax_pred = axes[j][0]
        ax_err  = axes[j][1]

        gt = results[model_names[0]][tname]['gt']
        x  = np.arange(len(gt))

        # 左：GT vs 各模型 Pred
        ax_pred.plot(x, gt, 'k--', linewidth=2, label='GT', zorder=10)
        for i, mname in enumerate(model_names):
            pred = results[mname][tname]['pred']
            ax_pred.plot(x, pred, color=colors[i], linewidth=1.5,
                         marker='o', markersize=4, label=mname)
        ax_pred.set_title(f'{tname} — GT vs Pred (LOO-CV)', fontsize=10)
        ax_pred.set_xlabel('Sample index')
        ax_pred.set_ylabel(tname)
        ax_pred.legend(fontsize=7, ncol=2)
        ax_pred.grid(True, alpha=0.3)

        # 右：各模型 MAE 柱状图
        maes = [results[m][tname]['mae'] for m in model_names]
        bars = ax_err.bar(model_names, maes, color=colors)
        ax_err.set_title(f'{tname} — MAE Comparison', fontsize=10)
        ax_err.set_ylabel('MAE')
        ax_err.tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, maes):
            ax_err.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        ax_err.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=150)
    print(f"Plot saved to: {out_img}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    data_root = os.path.join(PROJECT_ROOT, 'data')
    out_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'logs_ml')
    out_txt   = os.path.join(out_dir, 'ml_results.txt')
    out_img   = os.path.join(out_dir, 'ml_prediction_curves.png')

    print("Loading data ...")
    X, Y, feature_names, target_names = load_data(data_root)
    print(f"X shape: {X.shape},  Y shape: {Y.shape}")
    print(f"Features: {feature_names}")
    print(f"Targets : {target_names}\n")

    print("Running LOO-CV for all models ...")
    results = evaluate_loo(X, Y, target_names)

    print_results(results, target_names, out_txt)
    plot_results(results, target_names, out_img)


if __name__ == '__main__':
    main()
