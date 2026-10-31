import os
import re
import urllib.request

# 你的 NDBC 数据总目录
NDBC_BASE_DIR = r"F:\GNSS-R\DATA\E-data\NDBC-data"

# 你的黄金赛道边界 [North, West, South, East]
BOUND_N = 45.0
BOUND_W = -98.0
BOUND_S = 20.0
BOUND_E = -60.0


def clean_out_of_bounds_buoys():
    print("⏳ 正在下载 NOAA 全球浮标台站坐标簿...")
    url = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').splitlines()
    except Exception as e:
        print(f"❌ 下载站点元数据失败: {e}")
        return

    # 解析坐标字典
    buoy_locations = {}
    loc_pattern = re.compile(r'([\d\.]+)\s*([NS])\s+([\d\.]+)\s*([EW])')
    for line in lines:
        if line.startswith('#'): continue
        parts = line.split('|')
        if len(parts) >= 7:
            stn_id = parts[0].strip().lower()
            match = loc_pattern.search(parts[6].strip())
            if match:
                lat = float(match.group(1)) if match.group(2) == 'N' else -float(match.group(1))
                lon = float(match.group(3)) if match.group(4) == 'E' else -float(match.group(3))
                buoy_locations[stn_id] = (lat, lon)

    # 遍历本地文件夹，执行清洗
    deleted_count = 0
    kept_count = set()

    print("🧹 开始扫描并清理越界浮标文件...")
    for root, dirs, files in os.walk(NDBC_BASE_DIR):
        for filename in files:
            if filename.endswith(".txt.gz"):
                # 提取浮标 ID
                match = re.match(r'^([a-zA-Z0-9]+)h\d{4}', filename)
                if match:
                    stn_id = match.group(1).lower()

                    # 检查是否在边界框内
                    if stn_id in buoy_locations:
                        lat, lon = buoy_locations[stn_id]
                        if (BOUND_S <= lat <= BOUND_N) and (BOUND_W <= lon <= BOUND_E):
                            kept_count.add(stn_id)  # 在框内，保留
                            continue

                    # 不在框内，或者找不到坐标的，直接删除！
                    file_path = os.path.join(root, filename)
                    os.remove(file_path)
                    deleted_count += 1

    print("\n" + "=" * 50)
    print("🎉 清洗任务圆满完成！")
    print(f"🗑️ 共删除了 {deleted_count} 个非东海岸/墨西哥湾的冗余文件。")
    print(f"💎 你的黄金赛道内，最终保留了 {len(kept_count)} 个高质量核心浮标站点！")
    print("=" * 50)


if __name__ == "__main__":
    clean_out_of_bounds_buoys()