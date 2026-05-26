import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def prepare_data(dynamic_path, static_path):
    print("-> [监控] 1. 开始读取动态数据 (Parquet)...")
    # 强制使用 fastparquet 引擎，在 Windows 下更不容易发生内存溢出
    df_dyn = pd.read_parquet(dynamic_path, engine='fastparquet')

    print("-> [监控] 2. 开始读取静态数据 (CSV)...")
    df_sta = pd.read_csv(static_path)

    dyn_cols = ['ERA5_u10', 'ERA5_v10', 'ERA5_swh', 'ERA5_mwp', 'ERA5_blh', 'ERA5_sp', 'ERA5_t2m', 'ERA5_WSPD']
    sta_scalar_cols = ['DTC_km', 'Bathymetry_m']
    sta_fetch_cols = [col for col in df_sta.columns if 'fetch' in col.lower()]

    print("-> [监控] 3. 时间序列转换与排序...")
    df_dyn['Time_UTC'] = pd.to_datetime(df_dyn['Time_UTC'])
    df_dyn = df_dyn.sort_values(by=['Climate_ID', 'Time_UTC']).reset_index(drop=True)
    df_dyn['Station_Idx'] = pd.factorize(df_dyn['Climate_ID'])[0]

    print("-> [监控] 4. 执行三分切分...")
    train_mask = df_dyn['Time_UTC'].dt.year <= 2023
    val_mask = df_dyn['Time_UTC'].dt.year == 2024
    test_mask = df_dyn['Time_UTC'].dt.year >= 2025

    train_dyn = df_dyn[train_mask].copy().reset_index(drop=True)
    val_dyn = df_dyn[val_mask].copy().reset_index(drop=True)
    test_dyn = df_dyn[test_mask].copy().reset_index(drop=True)

    print(f"切分完成 -> Train: {len(train_dyn)} | Val: {len(val_dyn)} | Test: {len(test_dyn)}")

    print("-> [监控] 5. 备份物理真实风速...")
    train_dyn['ERA5_WSPD_ORIG'] = train_dyn['ERA5_WSPD'].copy()
    val_dyn['ERA5_WSPD_ORIG'] = val_dyn['ERA5_WSPD'].copy()
    test_dyn['ERA5_WSPD_ORIG'] = test_dyn['ERA5_WSPD'].copy()

    print("-> [监控] 6. 初始化 StandardScaler 并执行归一化...")
    scaler_dyn = StandardScaler()
    train_dyn[dyn_cols] = scaler_dyn.fit_transform(train_dyn[dyn_cols])
    val_dyn[dyn_cols] = scaler_dyn.transform(val_dyn[dyn_cols])
    test_dyn[dyn_cols] = scaler_dyn.transform(test_dyn[dyn_cols])

    print("-> [监控] 7. 处理静态空间特征...")
    scaler_sta_scalar = StandardScaler()
    df_sta[sta_scalar_cols] = scaler_sta_scalar.fit_transform(df_sta[sta_scalar_cols])
    scaler_sta_fetch = StandardScaler()
    df_sta[sta_fetch_cols] = scaler_sta_fetch.fit_transform(df_sta[sta_fetch_cols])
    df_sta.set_index('Climate_ID', inplace=True)

    print("-> [监控] 8. 数据准备彻底完成，准备送入 DataLoader！")
    return train_dyn, val_dyn, test_dyn, df_sta, dyn_cols, sta_scalar_cols, sta_fetch_cols


class TAMNetDataset(Dataset):
    def __init__(self, df_dyn, df_sta, dyn_cols, sta_scalar_cols, sta_fetch_cols, config):
        self.seq_len = config['seq_len']
        self.mode = config.get('correction_mode', 'residual')
        self.sta_scalar_cols = sta_scalar_cols
        self.sta_fetch_cols = sta_fetch_cols
        self.df_sta = df_sta

        self.dyn_values = df_dyn[dyn_cols].values.astype(np.float32)
        self.wspd_true = df_dyn['WSPD_10m'].values.astype(np.float32)

        # 【核心修复】：指向未归一化的物理风速列
        self.wspd_era5 = df_dyn['ERA5_WSPD_ORIG'].values.astype(np.float32)

        self.climate_ids = df_dyn['Climate_ID'].values
        self.station_idx = df_dyn['Station_Idx'].values

        self.valid_indices = []

        print(f"校验时序连续性与缺失值 (seq_len={self.seq_len})...")
        grouped = df_dyn.groupby('Climate_ID')

        for _, group in grouped:
            group_start_idx = group.index[0]
            group_length = len(group)
            times = group['Time_UTC'].values

            # 使用备份列进行 NaN 检查
            has_nan = group[dyn_cols + ['WSPD_10m', 'ERA5_WSPD_ORIG']].isna().values.any(axis=1)

            if group_length > self.seq_len:
                for i in range(group_length - self.seq_len):
                    if has_nan[i: i + self.seq_len].any():
                        continue
                    time_delta = (times[i + self.seq_len - 1] - times[i]) / np.timedelta64(1, 'h')
                    if time_delta == self.seq_len - 1:
                        self.valid_indices.append(group_start_idx + i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len

        X_dyn = torch.tensor(self.dyn_values[start_idx: end_idx])

        u_true = self.wspd_true[end_idx - 1]
        u_era5 = self.wspd_era5[end_idx - 1]  # 此时这里是真正的物理风速

        current_station_id = self.climate_ids[end_idx - 1]
        station_idx_val = self.station_idx[end_idx - 1]

        if self.mode == 'residual':
            y_target = u_true - u_era5
        else:  # factor 模式
            # 现在 u_era5 是绝对风速，截断 0.5 具备了真实的物理防除零意义
            safe_era5 = max(u_era5, 0.5)
            raw_factor = u_true / safe_era5
            y_target = np.clip(raw_factor, 0.2, 5.0)

        Y = torch.tensor([y_target], dtype=torch.float32)
        U_era5_tensor = torch.tensor([u_era5], dtype=torch.float32)
        U_true_tensor = torch.tensor([u_true], dtype=torch.float32)
        Station_Idx_tensor = torch.tensor([station_idx_val], dtype=torch.long)

        station_sta_data = self.df_sta.loc[current_station_id]
        X_sta_scalar = torch.tensor(station_sta_data[self.sta_scalar_cols].values.astype(np.float32))
        X_sta_fetch = torch.tensor(station_sta_data[self.sta_fetch_cols].values.astype(np.float32))

        return X_dyn, X_sta_scalar, X_sta_fetch, Y, U_era5_tensor, U_true_tensor, Station_Idx_tensor