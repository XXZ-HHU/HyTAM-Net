import os
import pandas as pd
import numpy as np
import requests
import re
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')  # 忽略 pandas 的部分链式赋值警告

# ================= 配置区 =================
PARQUET_ROOT_DIR = r"F:\GNSS-R\DATA\E-data\NDBC_PARQUET"
OUTPUT_CLEAN_DIR = r"F:\GNSS-R\DATA\E-data\NDBC_CLEANED_PARQUET"
STATION_META_OUTPUT = r"F:\GNSS-R\DATA\E-data\ndbc_station_metadata.csv"

# 质量控制参数
MIN_ANNUAL_VALID_RATE = 0.80  # 全年数据有效率需 >= 80%
MAX_INTERPOLATE_HOURS = 2  # 最大允许插值的断档时间 (2小时)
MIN_DISTANCE_KM = 1.0  # 站点去重的最小空间距离阈值


# ==========================================

# --- 模块 1: 空间计算与坐标抓取 ---
def haversine(lon1, lat1, lon2, lat2):
    """计算地球表面两经纬度点之间的球面距离 (公里)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径 (公里)
    return c * r


def fetch_station_coords(station_id):
    """从 NDBC 官网抓取站点的经纬度"""
    url = f"https://www.ndbc.noaa.gov/station_page.php?station={station_id}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text()
            # 匹配例如: 34.123 N 77.456 W (支持小数和度分秒转换较复杂，这里抓取十进制)
            # NDBC 页面通常在头部有: 34.300 N 77.370 W (34°18'0" N 77°22'12" W)
            match = re.search(r'(\d+\.\d+)\s*([NS])\s+(\d+\.\d+)\s*([EW])', page_text)
            if match:
                lat = float(match.group(1)) * (1 if match.group(2) == 'N' else -1)
                lon = float(match.group(3)) * (1 if match.group(4) == 'E' else -1)
                return lat, lon
    except:
        pass
    return None, None


# --- 模块 2: 时序处理核心 ---
def process_temporal_qc(df, year):
    """执行 80% 检查、时序缝合与熔断切片"""
    df['Date'] = pd.to_datetime(df['Date'])
    # 剔除同一小时的重复记录
    df = df.drop_duplicates(subset=['Date']).set_index('Date')

    # 1. 强制生成该年完整的 8760 小时时间轴
    start_date = f'{year}-01-01 00:00:00'
    end_date = f'{year}-12-31 23:00:00'
    full_idx = pd.date_range(start=start_date, end=end_date, freq='1H')
    df = df.reindex(full_idx)

    total_hours = len(full_idx)

    # 2. 检查 80% 可用率 (MOST 计算所需的三变量)
    valid_count = df[['WSPD', 'ATMP', 'WTMP']].dropna().shape[0]
    if (valid_count / total_hours) < MIN_ANNUAL_VALID_RATE:
        return None, 0  # 不达标，直接遗弃

    # 3. 小断层缝合 (只插值连续缺失 <= 2 小时的数据)
    # limit=2 确保了如果缺失 3 小时，这 3 小时都保持 NaN，不会被错误填补
    cols_to_fill = ['WSPD', 'ATMP', 'WTMP', 'WDIR']
    df[cols_to_fill] = df[cols_to_fill].interpolate(method='linear', limit=MAX_INTERPOLATE_HOURS)

    # 4. 大断层熔断 (生成 Segment_ID)
    # 此时依然为 NaN 的行，就是超过 2 小时的真实断档
    is_missing = df[['WSPD', 'ATMP', 'WTMP']].isna().any(axis=1)

    # 巧妙逻辑：每次遇到 NaN，cumsum 就会 +1，从而产生新的片段 ID
    # 这样后续网络读取时，只要按 Segment_ID 分组，组内的时间绝对是完美的 1小时 连续步长
    df['Segment_ID'] = is_missing.cumsum()

    # 清理掉真正的断档行，重置索引
    df_clean = df.dropna(subset=['WSPD', 'ATMP', 'WTMP']).reset_index()
    df_clean = df_clean.rename(columns={'index': 'Date'})
    df_clean['Year'] = int(year)

    return df_clean, len(df_clean)


# --- 主控引擎 ---
def execute_master_pipeline():
    os.makedirs(OUTPUT_CLEAN_DIR, exist_ok=True)
    year_folders = [d for d in os.listdir(PARQUET_ROOT_DIR) if d.startswith('Year=')]
    year_folders.sort()

    all_station_meta = {}  # 记录站点经纬度和总记录数用于去重
    cleaned_data_buffers = []

    print(" 阶段 1: 扫描并执行时序质控 (Temporal QC)...")
    for year_dir in tqdm(year_folders):
        year = year_dir.split('=')[1]
        file_path = os.path.join(PARQUET_ROOT_DIR, year_dir, 'data.parquet')

        if not os.path.exists(file_path): continue

        df_raw = pd.read_parquet(file_path, engine='pyarrow')
        grouped = df_raw.groupby('Climate_ID')

        for stn_id, group in grouped:
            # 传入核心时序处理函数
            df_clean, valid_len = process_temporal_qc(group, year)

            if df_clean is not None:
                # 标记该站点的有效数据
                df_clean['Climate_ID'] = stn_id
                cleaned_data_buffers.append(df_clean)

                # 记录站点总量，用于后续重叠时的“优胜劣汰”
                if stn_id not in all_station_meta:
                    all_station_meta[stn_id] = {'Total_Records': 0, 'Lat': None, 'Lon': None}
                all_station_meta[stn_id]['Total_Records'] += valid_len

    if not cleaned_data_buffers:
        print(" 没有数据通过 80% 可用率测试！请检查数据源或放宽阈值。")
        return

    # 合并所有初步清洗的数据
    master_df = pd.concat(cleaned_data_buffers, ignore_index=True)

    print("\n 阶段 2: 执行空间 1km 去重 (Spatial Collocation)...")
    unique_stations = list(all_station_meta.keys())
    print(f" 共有 {len(unique_stations)} 个站点通过了时序筛选，开始获取坐标...")

    for stn in tqdm(unique_stations, desc="坐标抓取"):
        lat, lon = fetch_station_coords(stn)
        all_station_meta[stn]['Lat'] = lat
        all_station_meta[stn]['Lon'] = lon

    # 转换为 DataFrame 用于距离矩阵计算
    meta_df = pd.DataFrame.from_dict(all_station_meta, orient='index').reset_index()
    meta_df.columns = ['Climate_ID', 'Total_Records', 'Lat', 'Lon']
    meta_df = meta_df.dropna(subset=['Lat', 'Lon'])  # 丢弃无坐标站点

    # 计算距离并打标剔除 (保留 Total_Records 最大的)
    stations_to_drop = set()
    for i, row1 in meta_df.iterrows():
        if row1['Climate_ID'] in stations_to_drop: continue

        for j, row2 in meta_df.iterrows():
            if i >= j or row2['Climate_ID'] in stations_to_drop: continue

            dist = haversine(row1['Lon'], row1['Lat'], row2['Lon'], row2['Lat'])
            if dist < MIN_DISTANCE_KM:
                # 发现 < 1km 的重叠，保留记录长的，剔除短的
                if row1['Total_Records'] >= row2['Total_Records']:
                    stations_to_drop.add(row2['Climate_ID'])
                else:
                    stations_to_drop.add(row1['Climate_ID'])
                    break  # row1 被干掉了，不需要再跟其他比了

    final_stations = meta_df[~meta_df['Climate_ID'].isin(stations_to_drop)]
    final_stations.to_csv(STATION_META_OUTPUT, index=False)

    print(f"\n 空间去重完成！剔除了 {len(stations_to_drop)} 个重叠或无坐标站点。")
    print(f" 最终保留的高质量独立站点数: {len(final_stations)}")

    print("\n 阶段 3: 导出最终纯净数据集...")
    # 只保留最终白名单站点的数据
    master_df_final = master_df[master_df['Climate_ID'].isin(final_stations['Climate_ID'])]

    # 按年份分区存储，优化后续加载速度
    for year, group in master_df_final.groupby('Year'):
        out_file = os.path.join(OUTPUT_CLEAN_DIR, f"NDBC_Cleaned_{year}.parquet")
        group.to_parquet(out_file, engine='pyarrow', index=False)
        print(f"   - 保存 {year} 年数据: {len(group):,} 行 -> {out_file}")

    print("\n" + "=" * 50)
    print(" NDBC 数据处理阶段 (1.1 & 1.2) 彻底完成！")
    print(f"数据现在已经完全具备时序连续性 (Segment_ID) 和空间独立性 (> 1km)。")
    print("=" * 50)


if __name__ == '__main__':
    execute_master_pipeline()