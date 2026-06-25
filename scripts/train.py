# scripts/train.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from configs.config import TAPMConfig
from src.dataset import BatteryTAPMDataset
from src.models import TAPMNetModel


def train_model(cfg, train_loader):
    model = TAPMNetModel(cfg).to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print("开始训练三通道自适应物理调制网络...")
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            h_soh = batch['x_hist_soh'].to(cfg.device)
            h_pos = batch['x_hist_pos'].to(cfg.device)
            h_neg = batch['x_hist_neg'].to(cfg.device)
            x_stat = batch['x_static'].to(cfg.device)
            k_pos = batch['k_future'].to(cfg.device)
            y_true = batch['y_future'].to(cfg.device)

            optimizer.zero_grad()
            y_pred = model(h_soh, h_pos, h_neg, x_stat, k_pos)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{cfg.epochs}] | 均方误差损失: {epoch_loss / len(train_loader):.6f}")
    return model


def evaluate_and_predict_eol(cfg, model, test_dataset, eol_threshold=80.0):
    model.eval()
    cells_data = {}

    for sample in test_dataset:
        c_id = sample['cell_id']
        if c_id not in cells_data:
            cells_data[c_id] = {
                'x_hist_soh': sample['x_hist_soh'].unsqueeze(0).to(cfg.device),
                'x_hist_pos': sample['x_hist_pos'].unsqueeze(0).to(cfg.device),
                'x_hist_neg': sample['x_hist_neg'].unsqueeze(0).to(cfg.device),
                'x_static': sample['x_static'].unsqueeze(0).to(cfg.device),
                'true_soh_trajectory': [],
                'known_cycles': []
            }
        cells_data[c_id]['true_soh_trajectory'].append(sample['y_future'].item() * 100.0)
        cells_data[c_id]['known_cycles'].append(sample['raw_k'])

    print("\n" + "=" * 60 + "\n★ TAPM-Net 寿命前推预测与外推推演评估 \n" + "=" * 60)

    for cell_id, info in cells_data.items():
        sort_idx = np.argsort(info['known_cycles'])
        known_cycles = np.array(info['known_cycles'])[sort_idx]
        true_soh = np.array(info['true_soh_trajectory'])[sort_idx]

        with torch.no_grad():
            z_soh = model.soh_encoder(info['x_hist_soh'])
            z_pos = model.pos_encoder(info['x_hist_pos'])
            z_neg = model.neg_encoder(info['x_hist_neg'])
            soh_g, soh_b, pos_g, pos_b, neg_g, neg_b = model.film(info['x_static'])

            z_soh_m = soh_g * z_soh + soh_b
            z_pos_m = pos_g * z_pos + pos_b
            z_neg_m = neg_g * z_neg + neg_b

        pred_soh_known = []
        with torch.no_grad():
            for cycle in known_cycles:
                k_tensor = torch.tensor([[cycle / 1000.0]], dtype=torch.float32).to(cfg.device)
                pred_val = model.decoder(z_soh_m, z_pos_m, z_neg_m, k_tensor).item() * 100.0
                pred_soh_known.append(pred_val)
        pred_soh_known = np.array(pred_soh_known)

        mae = mean_absolute_error(true_soh, pred_soh_known)
        rmse = np.sqrt(mean_squared_error(true_soh, pred_soh_known))
        r2 = r2_score(true_soh, pred_soh_known)

        # 纯外推预测寿命终止
        projected_cycles = []
        projected_soh = []
        pred_eol_cycle = None

        for k in range(cfg.history_window + 1, 2000):
            with torch.no_grad():
                k_tensor = torch.tensor([[k / 1000.0]], dtype=torch.float32).to(cfg.device)
                pred_val = model.decoder(z_soh_m, z_pos_m, z_neg_m, k_tensor).item() * 100.0

            projected_cycles.append(k)
            projected_soh.append(pred_val)

            if pred_val <= eol_threshold and pred_eol_cycle is None:
                pred_eol_cycle = k
                max_trend_cycle = k + 100
            if pred_eol_cycle is not None and k >= max_trend_cycle:
                break

        true_eol_idx = np.where(true_soh <= eol_threshold)[0]
        true_eol_cycle = known_cycles[true_eol_idx[0]] if len(true_eol_idx) > 0 else None

        print(f"电池 ID: {cell_id}")
        print(f"  [测试评估]: MAE: {mae:.3f}% | RMSE: {rmse:.3f}% | R²: {r2:.4f}")
        if true_eol_cycle:
            print(f"  真实 EOL: {true_eol_cycle} 圈")
        if pred_eol_cycle:
            print(f"  预测 EOL: {pred_eol_cycle} 圈")
            if true_eol_cycle:
                print(f"  绝对误差: {pred_eol_cycle - true_eol_cycle:+} 圈")
        print("-" * 60)

        plt.figure(figsize=(10, 5))
        plt.plot(known_cycles, true_soh, label='实际测量 SOH', color='black', linestyle='--',linewidth=1.2)
        plt.plot(projected_cycles, projected_soh, label='TAPM-Net 连续外推趋势', color='red', linewidth=1.5)
        plt.axhline(eol_threshold, color='gray', linestyle=':', label=f'EOL 阈值 ({eol_threshold}%)')
        if pred_eol_cycle:
            plt.scatter(pred_eol_cycle, eol_threshold, color='red', s=80, zorder=5,
                        label=f'预测 EOL: {pred_eol_cycle} 圈')
        if true_eol_cycle:
            plt.scatter(true_eol_cycle, eol_threshold, color='black', s=80, zorder=5, marker='x',
                        label=f'实际 EOL: {true_eol_cycle} 圈')
        plt.title(f'Continuous Lifespan Projection & EOL Prognostics - {cell_id}')
        plt.xlabel('Cycle Number')
        plt.ylabel('State of Health (SOH, %)')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(f"{cell_id}_prediction.png", dpi=300)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练 TAPM-Net 预测模型")
    parser.add_argument('--csv_path', type=str, default='data/processed/TJU_NCA_cleaned_dataset.csv',
                        help='数据清洗后的CSV文件路径')
    args = parser.parse_args()

    cfg = TAPMConfig()
    cfg.cleaned_csv_path = args.csv_path

    if not os.path.exists(cfg.cleaned_csv_path):
        print(f"未找到清洗后的数据集: {cfg.cleaned_csv_path}，请先运行数据清洗流水线。")
        sys.exit(1)

    df_clean = pd.read_csv(cfg.cleaned_csv_path)
    all_cells = list(df_clean['cell_id'].unique())
    cfg.train_cells = [c for c in all_cells if c not in cfg.test_cells]

    print(f"读取数据集成功：训练包含 {len(cfg.train_cells)} 块电池，测试验证包含 {len(cfg.test_cells)} 块电池。")

    train_dataset = BatteryTAPMDataset(df_clean, cfg, is_train=True)
    test_dataset = BatteryTAPMDataset(df_clean, cfg, is_train=False, scaler=train_dataset.scaler)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    trained_model = train_model(cfg, train_loader)
    evaluate_and_predict_eol(cfg, trained_model, test_dataset)

    # 保存模型
    checkpoint = {
        'model_state_dict': trained_model.state_dict(),
        'scaler_mean': train_dataset.scaler.mean_,
        'scaler_scale': train_dataset.scaler.scale_,
        'history_window': cfg.history_window,
        'latent_dim': cfg.latent_dim,
        'pos_exogenous_cols': cfg.pos_exogenous_cols,
        'neg_exogenous_cols': cfg.neg_exogenous_cols,
        'static_cols': cfg.static_cols
    }
    torch.save(checkpoint, cfg.model_save_path)
    print(f"模型和缩放器已成功保存至 {cfg.model_save_path}")