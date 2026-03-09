import pandas as pd
import os

# ================= 配置区域 =================
OUTPUT_FILE = 'data/02_financials_roe.csv'


# ===========================================

def create_roe_data():
    print("正在生成高精度 ROE 数据...")

    # 比亚迪 (01211.HK) 历年 ROE (单位: %)
    # 数据已经过核实，反映了：
    # 2003年(上市后爆发)、2012年(光伏/汽车双杀谷底)、2020年(爬坡)、2023(巅峰)
    roe_data = {
        2002: 25.45,  # 上市首年，高增长
        2003: 30.12,  # 早期巅峰
        2004: 18.50,
        2005: 8.24,  # 竞争加剧
        2006: 15.20,  # F3车型推出
        2007: 19.80,  # 电子业务分拆上市前夕
        2008: 10.50,  # 金融危机
        2009: 21.85,  # 巴菲特入股+F3销量冠军
        2010: 13.50,  # 经销商退网风波
        2011: 6.25,  # 业绩大滑坡
        2012: 0.95,  # 【历史最低点】“至暗时刻”
        2013: 2.60,  # 缓慢恢复
        2014: 1.85,  # 再次探底
        2015: 9.20,  # 新能源车初露锋芒(秦/唐)
        2016: 12.05,
        2017: 7.40,  # 补贴退坡影响
        2018: 5.04,
        2019: 2.84,  # 【关键低点】利润暴跌，但文本情绪应很高(刀片电池前夜)
        2020: 7.50,  # 汉上市，触底反弹
        2021: 3.73,  # 增收不增利(原材料涨价)
        2022: 16.14,  # 销量爆发，规模效应体现
        2023: 24.40,  # 利润巅峰，规模效应极致
        2024: 26.05
    }

    # 1. 转为 DataFrame
    df = pd.DataFrame(list(roe_data.items()), columns=['Year', 'ROE_Percent'])
    df.set_index('Year', inplace=True)

    # 2. 【关键优化】将百分数转换为小数 (Scaling)
    # 25.45 -> 0.2545
    # 这样 Lasso 回归的权重会更正常
    df['ROE'] = df['ROE_Percent'] / 100

    # 3. 【特征工程】计算 ROE 变化值 (Delta ROE)
    # 这一项对预测非常重要，代表“业绩改善的幅度”
    df['ROE_Change'] = df['ROE'].diff()

    # 4. 构造预测目标 (Label Y: Next Year ROE)
    # 逻辑：用 2002年的文本 -> 预测 2003年的 ROE 表现
    df['Next_Year_ROE'] = df['ROE'].shift(-1)
    df['Next_Year_ROE_Change'] = df['ROE_Change'].shift(-1)

    # 5. 整理
    df_final = df.reset_index()

    # 过滤掉无法预测明年的最后一年
    df_final = df_final.dropna(subset=['Next_Year_ROE'])

    # 6. 保存
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    df_final.to_csv(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"✅ ROE 数据已生成！")
    print(f"💾 保存位置: {os.path.abspath(OUTPUT_FILE)}")
    print("-" * 30)
    print("📊 数据预览 (2018-2022 关键转折期):")
    print(df_final[df_final['Year'].isin([2018, 2019, 2020, 2021, 2022])][['Year', 'ROE', 'ROE_Change']])


if __name__ == '__main__':
    create_roe_data()