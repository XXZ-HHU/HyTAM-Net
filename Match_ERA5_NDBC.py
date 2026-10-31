import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm

# ================= 配置区 =================
# 输入：已经 MOST 外推的 NDBC 年度数据目录 (Parquet 格式)
NDBC_MOST_DIR = r"F:\GNSS-R\DATA\E-data\NDBC_MOST_10M"

# 输入：包含浮标经纬度的元数据表
META_FILE = r"F:\GNSS-R\DATA\E-data\ndbc_station_metadata_cropped.csv"

# 输入：ERA5 每月 NetCDF 文件目录
ERA5_DIR = r"F:\GNSS-R\DATA\E-data\ERA5_FINAL_NC"

# 输出：最终的时空匹配宽表 (Parquet)
OUTPUT_FILE = r"F:\GNSS-R\DATA\E-data\ERA5_NDBC_Matched_MOST.parquet"

# 需提取的 ERA5 变量 (确保 NC 文件中包含 u10 和 v10)
ERA5_VARS = ['u10', 'v10', 'swh', 'mwp', 'blh', 'sp', 't2m']


# ==========================================

def collocate_era5_to_ndbc_most():
    print(f"📖 正在加载已完成 MOST 外推的 NDBC 年度 Parquet 文件...")

    ndbc_files = glob.glob(os.path.join(NDBC_MOST_DIR, "*.parquet"))

    if not ndbc_files:
        print(f"❌ 在 {NDBC_MOST_DIR} 未找到 .parquet 文件。")
        return

    df_list = []
    for f in ndbc_files:
        try:
            df = pd.read_parquet(f)
            df_list.append(df)
        except Exception as e:
            print(f"❌ 读取 {f} 失败: {e}")

    df_ndbc = pd.concat(df_list, ignore_index=True)
    print(f"✅ 成功加载 {len(df_ndbc)} 条 NDBC 记录。")

    # --- 1. 修复时间列嗅探 ---
    possible_time_cols = ['Time', 'time', 'Datetime', 'datetime', 'Date', 'date', 'Date_Time', 'time_utc', 'TIME']
    for col in possible_time_cols:
        if col in df_ndbc.columns:
            df_ndbc.rename(columns={col: 'Time_UTC'}, inplace=True)
            print(f"🔄 已自动将时间列 '{col}' 重命名为 'Time_UTC'")
            break

    if 'Time_UTC' not in df_ndbc.columns:
        print("❌ 致命错误：仍未找到时间列，请检查表头。")
        return

    df_ndbc['Time_UTC'] = pd.to_datetime(df_ndbc['Time_UTC'])
    df_ndbc['YearMonth'] = df_ndbc['Time_UTC'].dt.to_period('M')

    # --- 2. 修复缺失经纬度的问题 ---
    if 'Lat' not in df_ndbc.columns and 'Latitude' not in df_ndbc.columns and 'lat' not in df_ndbc.columns:
        print(f"🔍 发现时序数据中没有经纬度列，正在从元数据表合并 Lat/Lon ...")
        try:
            meta_df = pd.read_csv(META_FILE)
            # 仅提取站号和经纬度，左连接拼接到时序表中
            df_ndbc = pd.merge(df_ndbc, meta_df[['Climate_ID', 'Lat', 'Lon']], on='Climate_ID', how='left')
            # 丢弃无法匹配经纬度的废弃站点数据
            df_ndbc.dropna(subset=['Lat', 'Lon'], inplace=True)
            print("✅ 经纬度合并完成！")
        except Exception as e:
            print(f"❌ 读取或合并元数据表 {META_FILE} 失败: {e}")
            return
    else:
        # 自适应已有的经纬度列名
        if 'Latitude' in df_ndbc.columns:
            df_ndbc.rename(columns={'Latitude': 'Lat', 'Longitude': 'Lon'}, inplace=True)
        elif 'lat' in df_ndbc.columns:
            df_ndbc.rename(columns={'lat': 'Lat', 'lon': 'Lon'}, inplace=True)

    # --- 3. 开始执行 ERA5 矢量化匹配 ---
    era5_files = glob.glob(os.path.join(ERA5_DIR, "*.nc"))
    matched_chunks = []

    grouped = df_ndbc.groupby('YearMonth')

    for ym, group in tqdm(grouped, desc="按月时空双线性插值进度"):
        year = str(ym.year)
        month = f"{ym.month:02d}"

        target_file = [f for f in era5_files if f"_{year}_{month}" in f]

        if not target_file:
            continue

        nc_path = target_file[0]

        try:
            ds = xr.open_dataset(nc_path)

            # --- 修复维度命名不一致的问题 (CDS API 更新导致) ---
            if 'valid_time' in ds.dims:
                ds = ds.rename({'valid_time': 'time'})
            # ----------------------------------------------------

            target_lats = xr.DataArray(group['Lat'].values, dims="points")
            target_lons = xr.DataArray(group['Lon'].values, dims="points")
            target_times = xr.DataArray(group['Time_UTC'].values, dims="points")

            # 双向插值 (空间双线性，时间线性)

            ds_interp = ds[ERA5_VARS].interp(
                latitude=target_lats,
                longitude=target_lons,
                time=target_times,
                method="linear"
            )

            for var in ERA5_VARS:
                group[f'ERA5_{var}'] = ds_interp[var].values

            # 合成 ERA5 矢量风速
            group['ERA5_WSPD'] = np.sqrt(group['ERA5_u10'] ** 2 + group['ERA5_v10'] ** 2)

            matched_chunks.append(group)
            ds.close()

        except Exception as e:
            print(f"\n❌ 处理 {nc_path} 时发生错误: {e}")
            continue

    if not matched_chunks:
        print("❌ 未成功匹配任何数据，请检查 ERA5 文件路径。")
        return

    final_df = pd.concat(matched_chunks, ignore_index=True)

    initial_len = len(final_df)
    final_df.dropna(subset=['ERA5_WSPD'], inplace=True)
    dropped_len = initial_len - len(final_df)

    final_df.drop(columns=['YearMonth'], inplace=True)

    final_df.to_parquet(OUTPUT_FILE, index=False)

    print("\n" + "=" * 50)
    print(f"✅ 时空匹配完成！")
    print(f" - 成功合并记录数: {len(final_df)}")
    print(f" - 剔除越界/缺失数: {dropped_len}")
    print(f"📁 最终动态特征矩阵已保存至:\n   {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == '__main__':
    collocate_era5_to_ndbc_most()