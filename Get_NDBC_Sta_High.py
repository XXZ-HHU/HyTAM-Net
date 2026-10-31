import os
import pandas as pd
import requests
import re
import time
from tqdm import tqdm
from bs4 import BeautifulSoup

# ================= 配置区 =================
WHITELIST_INPUT = r"F:\GNSS-R\DATA\E-data\continuous_whitelist.csv"
HEIGHT_DICT_OUTPUT = r"F:\GNSS-R\DATA\E-data\ndbc_station_heights.csv"


# ==========================================

def fetch_station_height(station_id):
    url = f"https://www.ndbc.noaa.gov/station_page.php?station={station_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, None

        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()

        # 提取 Site elevation (站点海拔)
        # 匹配规则: "Site elevation: sea level" 或 "Site elevation: 15.2 m above mean sea level"
        site_elev = 0.0
        elev_match = re.search(r'Site elevation:\s*(sea level|(-?\d+\.?\d*)\s*m)', page_text, re.IGNORECASE)
        if elev_match:
            if elev_match.group(1).lower() == 'sea level':
                site_elev = 0.0
            else:
                site_elev = float(elev_match.group(2))

        # 提取 Anemometer height (风速计高度)
        # 匹配规则: "Anemometer height: 5 m above site elevation" 或 "Anemometer height: 4.1 m"
        anem_height = None
        anem_match = re.search(r'Anemometer height:\s*(\d+\.?\d*)\s*m', page_text, re.IGNORECASE)
        if anem_match:
            anem_height = float(anem_match.group(1))

        if anem_height is not None:
            # 最终物理高度 = 站点海拔 + 测风仪相对高度
            total_height = site_elev + anem_height
            return round(total_height, 2), "Success"
        else:
            return None, "Missing Anemometer Height"

    except Exception as e:
        return None, f"Error: {str(e)}"


def build_height_dictionary():
    print(f"📖 正在读取白名单: {WHITELIST_INPUT}")
    df = pd.read_csv(WHITELIST_INPUT)
    unique_stations = df['Station_ID'].unique()
    print(f"🔍 发现 {len(unique_stations)} 个独立站点，开始联网爬取高度信息...")

    results = []

    # 使用 tqdm 显示进度条
    for stn in tqdm(unique_stations, desc="爬取进度"):
        height, status = fetch_station_height(stn)
        results.append({
            'Station_ID': stn,
            'Anemometer_Height_m': height,
            'Fetch_Status': status
        })
        # 礼貌性延时，防止被 NDBC 封 IP
        time.sleep(0.3)

    height_df = pd.DataFrame(results)
    height_df.to_csv(HEIGHT_DICT_OUTPUT, index=False)

    # 统计战况
    success_count = height_df['Anemometer_Height_m'].notna().sum()
    fail_count = len(height_df) - success_count

    print("\n" + "=" * 50)
    print("✅ 高度字典构建完毕！")
    print(f"🎯 成功获取高度的站点: {success_count} 个")
    print(f"❌ 缺失/获取失败的站点: {fail_count} 个")
    print(f"📁 字典已保存至: {HEIGHT_DICT_OUTPUT}")

    if fail_count > 0:
        print("\n⚠️ 以下站点未能自动获取高度 (需手动检查或剔除):")
        failed_df = height_df[height_df['Anemometer_Height_m'].isna()]
        # 仅打印前 10 个作为示例
        print(failed_df.head(10).to_string(index=False))
    print("=" * 50)


if __name__ == '__main__':
    build_height_dictionary()