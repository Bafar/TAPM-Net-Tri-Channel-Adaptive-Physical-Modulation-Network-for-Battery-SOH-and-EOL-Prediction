# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class BatteryTAPMDataset(Dataset):
    def __init__(self, df, cfg, is_train=True, scaler=None):
        self.cfg = cfg
        self.samples = []

        cells = cfg.train_cells if is_train else cfg.test_cells
        self.df_sub = df[df['cell_id'].isin(cells)].copy()

        if self.df_sub.empty:
            raise ValueError(f"未找到指定的电池数据，请检查电池 ID 配置。")

        self.all_dynamic_cols = cfg.pos_exogenous_cols + cfg.neg_exogenous_cols
        if is_train:
            self.scaler = StandardScaler()
            self.df_sub[self.all_dynamic_cols] = self.scaler.fit_transform(self.df_sub[self.all_dynamic_cols])
        else:
            self.scaler = scaler
            self.df_sub[self.all_dynamic_cols] = self.scaler.transform(self.df_sub[self.all_dynamic_cols])

        for cell_id, group in self.df_sub.groupby('cell_id'):
            group = group.sort_values('cycle').reset_index(drop=True)
            if len(group) <= cfg.history_window:
                continue

            hist_group = group.iloc[:cfg.history_window]
            x_hist_soh = hist_group[[cfg.target_col]].values / 100.0
            x_hist_pos = hist_group[cfg.pos_exogenous_cols].values
            x_hist_neg = hist_group[cfg.neg_exogenous_cols].values

            raw_static = hist_group[cfg.static_cols].iloc[0].values
            temp_scaled = (raw_static[0] - 25.0) / 20.0
            rate_scaled = (raw_static[1] - 0.25) / 0.75
            x_static = np.array([temp_scaled, rate_scaled])

            for idx in range(cfg.history_window, len(group)):
                target_row = group.iloc[idx]
                k_future = np.array([target_row['cycle'] / 1000.0])
                y_future = np.array([target_row[cfg.target_col] / 100.0])

                self.samples.append({
                    'x_hist_soh': torch.tensor(x_hist_soh, dtype=torch.float32),
                    'x_hist_pos': torch.tensor(x_hist_pos, dtype=torch.float32),
                    'x_hist_neg': torch.tensor(x_hist_neg, dtype=torch.float32),
                    'x_static': torch.tensor(x_static, dtype=torch.float32),
                    'k_future': torch.tensor(k_future, dtype=torch.float32),
                    'y_future': torch.tensor(y_future, dtype=torch.float32),
                    'cell_id': cell_id,
                    'raw_k': target_row['cycle']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]