import os
import requests
from bs4 import BeautifulSoup
import time

# ================= 配置参数 =================
BASE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
# 你的 F 盘目标路径
SAVE_BASE_DIR = r"F:\GNSS-R\DATA\E-data\NDBC-data"

# 【修改点】实验需求区间：从 2008 年到 2025 年 (包含 2025)
TARGET_YEARS = list(range(2008, 2026))

# 伪装请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ============================================

def download_ndbc_data():
    if not os.path.exists(SAVE_BASE_DIR):
        os.makedirs(SAVE_BASE_DIR)
        print(f"创建基础目录: {SAVE_BASE_DIR}")

    print("正在连接 NDBC 服务器获取数据列表，请稍候...")

    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        response = session.get(BASE_URL, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"无法访问 NDBC 目录，请检查网络: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    all_files = [link.get('href') for link in links if link.get('href')]

    for year in TARGET_YEARS:
        year_str = str(year)
        target_files = [f for f in all_files if f.endswith(f'h{year_str}.txt.gz')]

        total = len(target_files)
        if total == 0:
            print(f"警告：未找到 {year_str} 年的数据文件。")
            continue

        print(f"\n================ 发现 {total} 个站点的 {year_str} 年数据，开始下载 ================")

        year_dir = os.path.join(SAVE_BASE_DIR, f"Year={year_str}")
        os.makedirs(year_dir, exist_ok=True)

        for i, file_name in enumerate(target_files):
            file_url = f"{BASE_URL}{file_name}"
            save_path = os.path.join(year_dir, file_name)

            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                print(f"[{i + 1}/{total}] 文件已存在，跳过: {file_name}")
                continue

            print(f"[{i + 1}/{total}] 正在下载: {file_name}...")
            try:
                file_data = session.get(file_url, timeout=20, stream=True)
                file_data.raise_for_status()

                with open(save_path, 'wb') as f:
                    for chunk in file_data.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                time.sleep(0.1)

            except Exception as e:
                print(f"下载 {file_name} 失败: {e}")

    print("\n🎉 2008-2025 年的数据下载任务已全部完成！")


if __name__ == "__main__":
    download_ndbc_data()