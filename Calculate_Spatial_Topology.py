import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import pyproj
from tqdm import tqdm
import warnings

# 忽略 shapely 内部的一些不影响计算的拓扑警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================= 配置区 =================
STATION_META_FILE = r"F:\GNSS-R\DATA\E-data\ndbc_station_metadata.csv"
GSHHG_SHP_FILE = r"F:\GNSS-R\DATA\E-data\Coastline_SHP\GSHHS_shp\h\GSHHS_h_L1.shp"
OUTPUT_TOPOLOGY_FILE = r"F:\GNSS-R\DATA\E-data\ndbc_spatial_topology.csv"

MAX_FETCH_KM = 500.0
ANGLE_STEP = 10


# ==========================================

def create_azimuthal_projection(lat, lon):
    return pyproj.CRS(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")


def calculate_station_topology():
    print(f"📖 正在加载 GSHHG 海岸线数据: {GSHHG_SHP_FILE} ...")
    gdf_land = gpd.read_file(GSHHG_SHP_FILE)

    print(f"📖 正在加载 NDBC 浮标元数据: {STATION_META_FILE}")
    stations_df = pd.read_csv(STATION_META_FILE).dropna(subset=['Lat', 'Lon'])

    topology_results = []
    land_sindex = gdf_land.sindex

    for idx, row in tqdm(stations_df.iterrows(), total=len(stations_df), desc="拓扑计算进度"):
        stn_id = row['Climate_ID']
        lat, lon = row['Lat'], row['Lon']

        buoy_pt_wgs84 = Point(lon, lat)
        local_crs = create_azimuthal_projection(lat, lon)
        transformer_to_local = pyproj.Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        buoy_pt_local = Point(transformer_to_local.transform(lon, lat))

        possible_matches_index = list(land_sindex.intersection(buoy_pt_wgs84.buffer(5.0).bounds))
        possible_matches = gdf_land.iloc[possible_matches_index]

        if possible_matches.empty:
            min_dist_km = MAX_FETCH_KM
        else:
            local_land = possible_matches.copy()
            local_land.geometry = local_land.geometry.to_crs(local_crs)

            # 【修复拓扑失效】: 消除多边形自相交与尖刺
            local_land.geometry = local_land.geometry.buffer(0)

            distances = local_land.geometry.distance(buoy_pt_local)
            min_dist_km = min(distances.min() / 1000.0, MAX_FETCH_KM)

        stn_result = {
            'Climate_ID': stn_id,
            'DTC_km': round(min_dist_km, 2)
        }

        for angle in range(0, 360, ANGLE_STEP):
            math_angle = np.radians(90 - angle)
            end_x = MAX_FETCH_KM * 1000.0 * np.cos(math_angle)
            end_y = MAX_FETCH_KM * 1000.0 * np.sin(math_angle)
            ray = LineString([(0, 0), (end_x, end_y)])

            fetch_km = MAX_FETCH_KM

            if not possible_matches.empty:
                # 在确保拓扑合法的前提下执行相交运算
                intersections = local_land.geometry.intersection(ray)
                valid_intersections = intersections[~intersections.is_empty]

                if not valid_intersections.empty:
                    closest_dist = valid_intersections.distance(buoy_pt_local).min() / 1000.0
                    fetch_km = min(closest_dist, MAX_FETCH_KM)

            stn_result[f'Fetch_{angle}_km'] = round(fetch_km, 2)

        topology_results.append(stn_result)

    topology_df = pd.DataFrame(topology_results)
    topology_df.to_csv(OUTPUT_TOPOLOGY_FILE, index=False)

    print("\n" + "=" * 50)
    print("✅ 浮标静态空间拓扑特征计算完成！")
    print(f"📊 共处理 {len(topology_results)} 个浮标站点。")
    print(f"📁 拓扑字典已保存至: {OUTPUT_TOPOLOGY_FILE}")
    print("=" * 50)

    cols_to_show = ['Climate_ID', 'DTC_km', 'Fetch_0_km', 'Fetch_90_km', 'Fetch_180_km', 'Fetch_270_km']
    print(topology_df[cols_to_show].head(1).to_string(index=False))


if __name__ == '__main__':
    calculate_station_topology()