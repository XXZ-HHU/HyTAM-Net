import os
import pandas as pd
from tqdm import tqdm

# ================= 配置区 =================
# 指向你存放所有 Year=XXXX 文件夹的根目录
PARQUET_ROOT_DIR = r"F:\GNSS-R\DATA\E-data\NDBC_PARQUET"
# 最终输出的白名单文件路径
WHITELIST_OUTPUT = r"F:\GNSS-R\DATA\E-data\multiyear_whitelist.csv"

# 筛选标准：
# 1. 该站点在该年必须有至少 2000 条数据 (约3个月的记录量，保证最基本的时序长度)
MIN_RECORDS_PER_YEAR = 2000
# 2. 能够用于 MOST 计算的数据 (同时具备 WSPD, ATMP, WTMP) 占比必须大于 70%
MIN_MOST_VALID_RATE = 0.70


# ==========================================

def generate_multiyear_whitelist():
    print(f"🔍 开始全局扫描数据库: {PARQUET_ROOT_DIR}")

    # 获取所有的年份文件夹
    year_folders = [d for d in os.listdir(PARQUET_ROOT_DIR) if d.startswith('Year=')]
    year_folders.sort()

    if not year_folders:
        print("❌ 未找到任何 Year=XXXX 文件夹，请检查路径。")
        return

    all_stats = []

    for year_dir in tqdm(year_folders, desc="跨年度扫描进度"):
        year_val = year_dir.split('=')[1]
        file_path = os.path.join(PARQUET_ROOT_DIR, year_dir, 'data.parquet')

        if not os.path.exists(file_path):
            continue

        try:
            # 读取单年数据
            df = pd.read_parquet(file_path, engine='pyarrow')

            # 按站点进行分组统计
            grouped = df.groupby('Climate_ID')
            for stn_id, group in grouped:
                total_records = len(group)

                # 统计同时具备 WSPD, ATMP, WTMP 的有效行数
                valid_most_records = group[['WSPD', 'ATMP', 'WTMP']].dropna().shape[0]
                valid_rate = valid_most_records / total_records if total_records > 0 else 0

                # 判断是否通过白名单筛选
                is_valid = (total_records >= MIN_RECORDS_PER_YEAR) and (valid_rate >= MIN_MOST_VALID_RATE)

                all_stats.append({
                    'Station_ID': stn_id,
                    'Year': year_val,
                    'Total_Records': total_records,
                    'MOST_Valid_Records': valid_most_records,
                    'Valid_Rate': round(valid_rate, 3),
                    'Is_Usable': is_valid
                })
        except Exception as e:
            print(f"⚠️ 读取 {year_val} 年数据时出错: {e}")

    # 将统计结果转为 DataFrame 并保存
    stats_df = pd.DataFrame(all_stats)

    # 只提取通过测试的白名单记录
    whitelist_df = stats_df[stats_df['Is_Usable'] == True].copy()

    # 保存到 CSV
    whitelist_df.to_csv(WHITELIST_OUTPUT, index=False)

    print("\n" + "=" * 50)
    print("✅ 全局扫描完成！")
    print(f"📊 数据库共包含 {len(stats_df)} 个 站-年 组合。")
    print(f"🎯 经过物理条件筛选，共有 {len(whitelist_df)} 个 站-年 组合进入最终白名单。")
    unique_usable_stations = whitelist_df['Station_ID'].nunique()
    print(f"🆔 至少有一年数据可用的独立浮标总数: {unique_usable_stations} 个。")
    print(f"📁 白名单已保存至: {WHITELIST_OUTPUT}")
    print("=" * 50)


if __name__ == '__main__':
    generate_multiyear_whitelist()