import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from dataloader import prepare_data, TAMNetDataset
import model_config


# ==========================================
# 核心技巧：时空元数据反向组装 (与 TAM-Net 格式完全对齐)
# ==========================================
def align_metadata(test_df, sta_df_raw, station_idx_arr, era5_arr, true_arr, pred_arr):
    print("\n-> [数据重建] 正在反向映射空间元数据 (测站名, 经度, 纬度)...")

    # 1. 还原测站名 Climate_ID
    sta_map = test_df[['Station_Idx', 'Climate_ID']].drop_duplicates().set_index('Station_Idx')['Climate_ID'].to_dict()

    res_df = pd.DataFrame({
        'Station_Idx': station_idx_arr,
        'Climate_ID': [sta_map[idx] for idx in station_idx_arr],
        'Time_UTC': np.nan,  # 预留时间列
        'Lat': np.nan,  # 预留纬度
        'Lon': np.nan,  # 预留经度
        'ERA5_Wind': era5_arr,
        'XGBoost_Pred': pred_arr,  # 👈 这里改为了 XGBoost 的专属列名
        'True_Wind': true_arr
    })

    # 2. 拼接经纬度 (采用不区分大小写的自动侦测机制)
    lat_col = next((c for c in sta_df_raw.columns if c.lower() in ['lat', 'latitude']), None)
    lon_col = next((c for c in sta_df_raw.columns if c.lower() in ['lon', 'longitude']), None)

    if lat_col and lon_col:
        lat_map = sta_df_raw.set_index('Climate_ID')[lat_col].to_dict()
        lon_map = sta_df_raw.set_index('Climate_ID')[lon_col].to_dict()
        res_df['Lat'] = res_df['Climate_ID'].map(lat_map)
        res_df['Lon'] = res_df['Climate_ID'].map(lon_map)
        print(f"-> [成功] 识别到经纬度列: '{lat_col}', '{lon_col}' 并映射完毕。")
    else:
        print(f"⚠️ [警告] 未在静态表中找到经纬度列，将跳过经纬度映射。")

    print("-> [数据重建] 正在通过双指针算法精准还原时间戳 (Time_UTC)...")

    # 3. 严格对齐时间戳
    for sta_idx, group in tqdm(res_df.groupby('Station_Idx'), desc="Time Alignment"):
        orig_group = test_df[test_df['Station_Idx'] == sta_idx].reset_index()
        orig_era5 = orig_group['ERA5_WSPD_ORIG'].values
        orig_times = orig_group['Time_UTC'].values

        group_era5 = group['ERA5_Wind'].values

        p_orig, p_res = 0, 0
        while p_res < len(group_era5) and p_orig < len(orig_era5):
            if np.isclose(group_era5[p_res], orig_era5[p_orig], atol=1e-4):
                res_df.loc[group.index[p_res], 'Time_UTC'] = orig_times[p_orig]
                p_res += 1
            p_orig += 1

    print("-> 时空元数据组装完成！")

    final_cols = ['Time_UTC', 'Climate_ID', 'Lat', 'Lon', 'ERA5_Wind', 'XGBoost_Pred', 'True_Wind']
    return res_df[final_cols]


def extract_features_from_loader(loader, mode='residual'):
    """从 DataLoader 中提取数据并展平为 XGBoost 可用的二维格式"""
    X_list, Y_list = [], []
    U_era5_list, U_true_list, Station_list = [], [], []

    for X_dyn, X_sta_scalar, X_sta_fetch, Y, U_era5, U_true, Station_Idx in tqdm(loader, desc="提取特征"):
        # 展平动态特征: [Batch, Seq_len, Features] -> [Batch, Seq_len * Features]
        batch_size = X_dyn.shape[0]
        X_dyn_flat = X_dyn.view(batch_size, -1).numpy()
        X_sta_scalar_np = X_sta_scalar.numpy()
        X_sta_fetch_np = X_sta_fetch.numpy()

        # 拼接所有特征
        X_concat = np.concatenate([X_dyn_flat, X_sta_scalar_np, X_sta_fetch_np], axis=1)

        X_list.append(X_concat)
        Y_list.append(Y.numpy().flatten())
        U_era5_list.append(U_era5.numpy().flatten())
        U_true_list.append(U_true.numpy().flatten())
        Station_list.append(Station_Idx.numpy().flatten())

    return (np.concatenate(X_list), np.concatenate(Y_list),
            np.concatenate(U_era5_list), np.concatenate(U_true_list), np.concatenate(Station_list))


def main():
    DYN_PATH = r'G:\DeepLearning\TAM_NET\Data\ERA5_NDBC_Matched_MOST.parquet'
    STA_PATH = r'G:\DeepLearning\TAM_NET\Data\ndbc_spatial_topology_final.csv'

    with open('data_config.json', 'r') as f:
        data_config = json.load(f)
    config = data_config | model_config.TAMNet_Config
    correction_mode = config.get('correction_mode', 'residual')

    train_df, val_df, test_df, sta_df, dyn_cols, sta_scalar_cols, sta_fetch_cols = prepare_data(DYN_PATH, STA_PATH)

    # 提取静态原表用于经纬度匹配
    sta_df_raw = pd.read_csv(STA_PATH)

    # batch_size 设大一点加速数据提取
    train_loader = DataLoader(TAMNetDataset(train_df, sta_df, dyn_cols, sta_scalar_cols, sta_fetch_cols, config),
                              batch_size=512, num_workers=0)
    val_loader = DataLoader(TAMNetDataset(val_df, sta_df, dyn_cols, sta_scalar_cols, sta_fetch_cols, config),
                            batch_size=512, num_workers=0)
    test_loader = DataLoader(TAMNetDataset(test_df, sta_df, dyn_cols, sta_scalar_cols, sta_fetch_cols, config),
                             batch_size=512, num_workers=0)

    print("\n--- 构建 XGBoost 训练集 ---")
    X_train, Y_train, _, _, _ = extract_features_from_loader(train_loader, correction_mode)
    print("--- 构建 XGBoost 验证集 ---")
    X_val, Y_val, _, _, _ = extract_features_from_loader(val_loader, correction_mode)
    print("--- 构建 XGBoost 测试集 ---")
    X_test, Y_test, U_era5_test, U_true_test, Station_test = extract_features_from_loader(test_loader, correction_mode)

    print("\n初始化 XGBoost 模型并开始训练 (可能需要一些时间)...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',  # 自动调用 GPU
        device='cuda',  # 强制使用 CUDA
        n_jobs=-1,
        early_stopping_rounds=15
    )

    xgb_model.fit(
        X_train, Y_train,
        eval_set=[(X_val, Y_val)],
        verbose=True
    )

    print("\n训练完成，开始预测 2025 年测试集...")
    Y_pred = xgb_model.predict(X_test)

    # 还原物理风速
    if correction_mode == 'residual':
        pred_wind = U_era5_test + Y_pred
    else:
        pred_wind = U_era5_test * Y_pred

    # ==================================================
    # 调用对齐算法，生成带时空信息的富格式结果表
    # ==================================================
    rich_result_df = align_metadata(
        test_df,
        sta_df_raw,
        Station_test.astype(int),
        U_era5_test,
        U_true_test,
        pred_wind
    )

    save_name = 'test_predictions_2025_xgboost_rich.csv'
    rich_result_df.to_csv(save_name, index=False)
    print(f"\n✅ XGBoost 预测明细 (含时空坐标) 已保存至: {save_name}")

    # ==================================================
    # 计算评估指标 (使用原始逻辑以保证结果一致)
    # ==================================================
    test_result_df = pd.DataFrame({
        'station': Station_test,
        'pred': pred_wind,
        'true': U_true_test
    })

    def calc_metrics(df):
        mae = np.mean(np.abs(df['pred'] - df['true']))
        rmse = np.sqrt(np.mean((df['pred'] - df['true']) ** 2))
        mbe = np.mean(df['pred'] - df['true'])
        ss_res = np.sum((df['true'] - df['pred']) ** 2)
        ss_tot = np.sum((df['true'] - df['true'].mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return pd.Series({'MAE': mae, 'RMSE': rmse, 'MBE': mbe, 'R2': r2})

    test_temporal_metrics = test_result_df.groupby('station').apply(calc_metrics, include_groups=False)
    tt_mae = test_temporal_metrics['MAE'].median()
    tt_rmse = test_temporal_metrics['RMSE'].median()
    tt_mbe = test_temporal_metrics['MBE'].median()
    tt_r2 = test_temporal_metrics['R2'].median()

    test_station_means = test_result_df.groupby('station')[['pred', 'true']].mean()
    tc_mae = np.mean(np.abs(test_station_means['pred'] - test_station_means['true']))
    tc_rmse = np.sqrt(np.mean((test_station_means['pred'] - test_station_means['true']) ** 2))
    tc_mbe = np.mean(test_station_means['pred'] - test_station_means['true'])
    tc_r2 = np.corrcoef(test_station_means['pred'], test_station_means['true'])[0, 1] ** 2 if len(
        test_station_means) > 1 else 0

    print("\n" + "★" * 50)
    print(f"      XGBoost BASELINE: TEST SET (2025) 精度报告")
    print("★" * 50)
    print(f"时序动态 (Temporal)  -> MAE: {tt_mae:.3f} | RMSE: {tt_rmse:.3f} | MBE: {tt_mbe:.3f} | R2: {tt_r2:.3f}")
    print(f"空间气候 (Climatology)-> MAE: {tc_mae:.3f} | RMSE: {tc_rmse:.3f} | MBE: {tc_mbe:.3f} | R2: {tc_r2:.3f}")
    print("★" * 50 + "\n")


if __name__ == '__main__':
    main()