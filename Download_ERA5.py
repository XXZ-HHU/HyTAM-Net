import cdsapi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
SAVE_DIR = r"F:\GNSS-R\DATA\E-data\ERA5_RAW_NC"
BOUNDING_BOX = [45.0, -98.0, 20.0, -60.0]

YEARS = [str(y) for y in range(2010, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'boundary_layer_height',
    'surface_pressure',
    '2m_temperature',
    'significant_height_of_combined_wind_waves_and_swell',
    'mean_wave_period'
]

# 【核心修改】最大并发数。CDS 官方一般允许同时进行 3-4 个任务。
# 设置为 4 可以最大化利用你的配额，同时防止被服务器判定为恶意攻击而封禁。
MAX_WORKERS = 4


# ==========================================

def download_single_month(year, month):
    """处理单个月份的下载任务"""
    file_name = f"ERA5_Coast_{year}_{month}.nc"
    file_path = os.path.join(SAVE_DIR, file_name)

    # 断点续传逻辑
    if os.path.exists(file_path) and os.path.getsize(file_path) > 100000:
        return f"[跳过] {file_name} 已存在。"

    print(f"🚀 提交请求: {year}年 {month}月 ...")
    c = cdsapi.Client()  # 每个线程需要独立的客户端实例

    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': VARIABLES,
                'year': year,
                'month': month,
                'day': [f"{d:02d}" for d in range(1, 32)],
                'time': [f"{t:02d}:00" for t in range(0, 24)],
                'area': BOUNDING_BOX,
            },
            file_path
        )
        return f"✅ 下载完成: {file_name}"
    except Exception as e:
        return f"❌ 下载 {file_name} 失败: {e}"


def main_fast_download():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 组装所有需要下载的任务清单
    tasks = []
    for year in YEARS:
        for month in MONTHS:
            tasks.append((year, month))

    print(f"总计 {len(tasks)} 个下载任务。启动 {MAX_WORKERS} 线程并发排队引擎...")

    # 2. 启动线程池进行并发请求
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 将任务提交给线程池
        future_to_task = {executor.submit(download_single_month, y, m): (y, m) for y, m in tasks}

        # as_completed 会在某个任务完成时立刻产出结果
        for future in as_completed(future_to_task):
            y, m = future_to_task[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"⚠️ {y}年{m}月 任务引发了异常: {exc}")


if __name__ == '__main__':
    main_fast_download()