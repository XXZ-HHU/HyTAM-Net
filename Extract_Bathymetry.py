import os
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm

# ================= 配置区 =================
# 你刚下载的 GEBCO NetCDF 文件路径
GEBCO_NC_FILE = r"F:\GNSS-R\DATA\E-data\gebco_2025_n65.0_s10.0_w-180.0_e-50.0.nc"

# 之前裁剪出的站点元数据 (包含 Lat, Lon)
META_FILE = r"F:\GNSS-R\DATA\E-data\ndbc_station_metadata_cropped.csv"
# 之前裁剪出的拓扑特征 (包含 DTC, Fetch)
TOPO_FILE = r"F:\GNSS-R\DATA\E-data\ndbc_spatial_topology_cropped.csv"

# 最终合并了水深的拓扑输出文件
OUTPUT_FINAL_TOPO = r"F:\GNSS-R\DATA\E-data\ndbc_spatial_topology_final.csv"


# ==========================================

def extract_station_bathymetry():
    print(f"📖 正在加载 GEBCO 水深网格: {os.path.basename(GEBCO_NC_FILE)} ...")
    try:
        ds_gebco = xr.open_dataset(GEBCO_NC_FILE)
    except Exception as e:
        print(f"❌ 读取 GEBCO 数据失败: {e}")
        return

    print("📖 正在加载站点元数据与拓扑矩阵...")
    meta_df = pd.read_csv(META_FILE)
    topo_df = pd.read_csv(TOPO_FILE)

    # 确保只处理水上站点 (DTC > 0)
    topo_df = topo_df[topo_df['DTC_km'] > 0.0].copy()

    # 合并以获取经纬度
    df_merged = pd.merge(topo_df, meta_df[['Climate_ID', 'Lat', 'Lon']], on='Climate_ID', how='left')

    bathymetry_results = []

    print("🌊 开始提取站点绝对水深...")
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="水深点抽样"):
        stn_id = row['Climate_ID']
        lat, lon = row['Lat'], row['Lon']

        try:
            # 经度对齐: NDBC 的西经是负数，GEBCO 通常也是 -180 到 180，直接匹配
            # 使用最邻近插值法 (nearest) 直接获取浮标正下方的海底高程
            elevation = float(ds_gebco.sel(lat=lat, lon=lon, method='nearest')['elevation'].values)

            # GEBCO 中，海平面以下为负值 (如 -1500m 表示水深 1500米)
            # 我们将其转换为正数的水深 (Bathymetry) 以符合物理直觉，若高程 > 0 则水深为 0
            water_depth = abs(elevation) if elevation < 0 else 0.0

            bathymetry_results.append({
                'Climate_ID': stn_id,
                'Elevation_m': round(elevation, 1),
                'Bathymetry_m': round(water_depth, 1)
            })

        except KeyError:
            # 如果浮标坐标超出了你下载的 GEBCO 边界框
            bathymetry_results.append({
                'Climate_ID': stn_id,
                'Elevation_m': np.nan,
                'Bathymetry_m': np.nan
            })

    ds_gebco.close()

    # 构建水深 DataFrame
    bathy_df = pd.DataFrame(bathymetry_results)

    # 将水深数据合并回原有的拓扑矩阵
    final_topo_df = pd.merge(topo_df, bathy_df[['Climate_ID', 'Bathymetry_m']], on='Climate_ID', how='left')

    # 调整列顺序，让 Bathymetry_m 紧跟在 DTC_km 后面
    cols = final_topo_df.columns.tolist()
    cols.insert(2, cols.pop(cols.index('Bathymetry_m')))
    final_topo_df = final_topo_df[cols]

    # 保存最终结果
    final_topo_df.to_csv(OUTPUT_FINAL_TOPO, index=False)

    print("\n" + "=" * 50)
    print("✅ 站点水深提取完毕！")
    print(f"📁 包含三维海岸地形约束的最终拓扑文件已保存至:\n   {OUTPUT_FINAL_TOPO}")
    print("=" * 50)

    # 打印前几个站点验证
    print("\n🔍 数据结果抽检 (前 5 个站点):")
    print(final_topo_df[['Climate_ID', 'DTC_km', 'Bathymetry_m']].head())


if __name__ == '__main__':
    extract_station_bathymetry()