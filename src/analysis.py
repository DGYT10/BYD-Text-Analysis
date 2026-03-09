import pandas as pd
import numpy as np
import jieba
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import statsmodels.api as sm
import platform

# ================= 0. 全局配置与路径 =================
# 设置绘图风格和字体，确保中文显示正常
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False

# 【路径智能适配】
# 无论脚本是在根目录还是在 src/ 目录下，都能找到 data 文件夹
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)

# 如果当前脚本在 src 文件夹中，说明根目录在上一级
if os.path.basename(project_root) == 'src':
    project_root = os.path.dirname(project_root)

# 定义数据路径
data_dir = os.path.join(project_root, 'data')
FILE_TEXT = os.path.join(data_dir, '01_text.csv')
FILE_ROE = os.path.join(data_dir, '02_financials_roe.csv')
FILE_FIN = os.path.join(data_dir, '03_fin.csv')
LOCAL_MODEL_PATH = os.path.join(data_dir, 'my_financial_bert')

# 定义输出图片路径
path_chart1 = os.path.join(data_dir, 'Chart_1_Strategy_River.png')
path_chart2 = os.path.join(data_dir, 'Chart_2_Mechanism_Transmission.png')
path_chart3 = os.path.join(data_dir, 'Chart_3_Correlation_Evidence.png')
path_chart4 = os.path.join(data_dir, 'Chart_4_Resilience_Score.png')
path_final_csv = os.path.join(data_dir, 'FINAL_ANALYSIS_TABLE.csv')

# ================= 1. 数据读取与合并 =================
print("1. 正在读取并合并数据...")
df_text = pd.read_csv(FILE_TEXT)
df_roe = pd.read_csv(FILE_ROE)
df_fin = pd.read_csv(FILE_FIN)

# 合并表格：基于年份连接 文本、ROE、财务数据
df = pd.merge(df_text, df_roe[['Year', 'ROE', 'Next_Year_ROE']], on='Year', how='inner')
df = pd.merge(df, df_fin[['Year', 'RDExpense_Billion', 'Revenue_Billion', 'CashFlow_Billion']], on='Year', how='inner')

# 构造核心变量：取对数以平滑数据分布，符合经济学实证惯例
df['Log_Revenue'] = np.log(df['Revenue_Billion']) # Y: 企业规模
df['Log_RD'] = np.log(df['RDExpense_Billion'])    # M: 研发投入
# 【新增】稳健性检验变量
df['Log_CashFlow'] = np.log(df['CashFlow_Billion'].apply(lambda x: x if x > 0 else 1)) # 防止负数报错

print(f"✅ 数据准备就绪，样本量: {len(df)}")

# ================= 2. NLP 模块一: 强力基础清洗 =================
print("2. 正在进行文本清洗...")

# A. 保护词典：防止专业术语被错误切分
tech_words = [
    '二次充电电池', '二次电池', '手机部件', '刀片电池', 'DM-i', '云轨', '易四方', '仰望',
    '方程豹', '云辇', 'e平台', '插电式混合动力', '磷酸铁锂', '三元锂', 'IGBT', '半导体',
    '储能', '王传福', '新能源', '电动车', '智能化', '垂直整合', '出海', '高端化',
    '乘用车', '商用车', '城市轨道交通'
]
for w in tech_words: jieba.add_word(w)

# B. 白名单：强制保留的单字或短词
whitelist = {'秦', '汉', '唐', '宋', '元', 'F3', 'F0', 'S6', 'e6', 'K9', '云', '芯', '二次', 'B2B'}

# C. 黑名单：剔除无意义的通用词和年报套话
base_stops = {'的', '了', '在', '是', '本集团', '公司', '我们', '年度', '增长', '发展', '业务', '及', '与', '为',
              '以及', '及其'}
report_stops = {
    '三十一日', '十二月', '三十日', '六月', '二零', '二零二', '二零一', '二零零',
    '全年', '期间', '期内', '截至', '年内', '回顾', '展望', '本年', '本年度',
    '二零零年', '二零零三年', '二零零四年', '二零二一年', '二零二二年',
    '归属于', '二年', '正式', '将会', '报告', '预期', '国内外', '本次',
    '人民币', '百万元', '千元', '亿元', '百万', '千万', '预计', '同比', '销量', '出货量',
    '实现', '进行', '装配', '领域', '能力', '产品', '客户', '市场', '行业', '集团',
    '部分', '项目', '技术', '收入', '成本', '增加', '减少', '相比', '保持', '录得',
    '表现', '上升', '助力', '疫情', '影响', '情况', '相关', '主要', '显示', '表示',
    '达到', '拥有', '提供', '包括', '位于', '约为', '综合', '由于', '因此'
}
stop_words = base_stops.union(report_stops)

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    words = jieba.lcut(text)
    clean_words = []
    for w in words:
        if w in whitelist:
            clean_words.append(w)
            continue
        if w in stop_words or w.isdigit() or len(w) < 2: continue
        if '二零' in w: continue
        clean_words.append(w)
    return " ".join(clean_words)

df['Clean_Text'] = df['Full_Text'].apply(clean_text)

# ================= 3. NLP 模块二: 双轨特征提取 =================

# --- 轨道 A: Financial-BERT (对照组 X2) ---
# 目的：捕捉通用的财务情绪（正面/负面），用于验证其对制造业战略预测的局限性
print("[Step 3A] 加载 Financial-BERT 模型...")
try:
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
    sentiment_pipeline = pipeline("sentiment-analysis", model=LOCAL_MODEL_PATH, tokenizer=LOCAL_MODEL_PATH,
                                  truncation=True, max_length=512, top_k=None, device=device)

    def get_bert_sentiment(text):
        if pd.isna(text) or len(str(text)) < 5: return 50.0
        text = str(text)
        chunk_size = 400
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)][:15]
        total_score = 0
        valid = 0
        try:
            results = sentiment_pipeline(chunks)
            for res in results:
                score_dict = {item['label']: item['score'] for item in res}
                # 兼容不同模型的标签命名
                pos = score_dict.get('LABEL_2', score_dict.get('正面', score_dict.get('Positive', 0)))
                neg = score_dict.get('LABEL_0', score_dict.get('负面', score_dict.get('Negative', 0)))
                total_score += (pos - neg)
                valid += 1
        except Exception:
            return 50.0
        if valid == 0: return 50.0
        # 归一化到 0-100 分
        return (total_score / valid + 1) / 2 * 100

    df['Financial_Sentiment'] = df['Clean_Text'].apply(get_bert_sentiment)
    print("✅ BERT 情感计算完成。")
except Exception as e:
    print(f"⚠️ BERT 模型加载失败或未训练 ({e})，使用 0 填充，不影响后续代码。")
    df['Financial_Sentiment'] = 0

# --- 轨道 B: 战略扩张指数 (核心变量 X1) ---
# 目的：基于特化词典捕捉管理层的“扩张意图”，这是驱动制造业增长的核心
print("[Step 3B] 计算战略扩张指数 (基于特化词典)...")
pos_expansion = {
    '增长', '增加', '提升', '上升', '新高', '突破', '翻番', '暴增', '领先', '跨越',
    '投产', '量产', '扩产', '交付', '建设', '竣工', '基地', '园区', '产能', '规模',
    '上市', '推出', '发布', '出海', '全球化', '渗透', '份额', '拓展', '中标', '签订',
    '研发', '创新', '专利', '独家', '首创', '技术', '新能源', '智能化', '高端', '领先'
}
neg_contraction = {
    '亏损', '损失', '减少', '下降', '下滑', '下跌', '缩减', '赤字', '摊薄', '减值',
    '风险', '压力', '挑战', '困难', '瓶颈', '疲软', '低迷', '放缓', '严峻',
    '停产', '延迟', '召回', '缺陷', '报废', '闲置', '违约', '诉讼'
}

def get_strategy_expansion_score(text):
    if pd.isna(text): return 0, 0
    words = str(text).split()
    if len(words) == 0: return 0, 0
    pos_cnt = sum(1 for w in words if w in pos_expansion)
    neg_cnt = sum(1 for w in words if w in neg_contraction)
    # 返回 (密度指数, 纯计数对数)
    density = (pos_cnt - neg_cnt) / len(words) * 1000
    log_count = np.log(pos_cnt + 1) # +1 防止 log(0)
    return density, log_count

# 应用函数并拆分结果
df[['Expansion_Index', 'Ln_Expansion_Count']] = df['Clean_Text'].apply(
    lambda x: pd.Series(get_strategy_expansion_score(x))
)

# ================= 4. LDA 主题建模 (生成 Topic 数据) =================
print("\n4. 运行 LDA 主题模型...")
tf_vectorizer = CountVectorizer(max_df=0.6, min_df=2, max_features=200)
tf_matrix = tf_vectorizer.fit_transform(df['Clean_Text'])

n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_output = lda.fit_transform(tf_matrix)

print("-" * 50)
print("LDA 主题关键词:")
feature_names = tf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    total_weight = topic.sum()
    top_indices = topic.argsort()[:-11:-1]
    top_features = [f"{feature_names[i]}({topic[i] / total_weight * 100:.1f}%)" for i in top_indices]
    print(f"Topic {topic_idx}: {', '.join(top_features)}")
print("-" * 50)

# 将 Topic 数据合并回主表
topic_cols = [f'Topic_{i}' for i in range(n_topics)]
df_topics = pd.DataFrame(lda_output, columns=topic_cols)
if 'Topic_0' in df.columns:
    df = df.drop(columns=topic_cols)
df = pd.concat([df, df_topics], axis=1)

# ================= 5. 实证回归 (机制检验升级版) =================
print("\n5. 运行多元回归验证...")

# 准备回归数据：剔除早期不稳定年份，标准化自变量
model_df = df[df['Year'] > 2005].copy().dropna(subset=['Expansion_Index', 'Financial_Sentiment', 'Log_Revenue', 'Log_RD', 'ROE', 'Log_CashFlow', 'Ln_Expansion_Count'])
scaler = StandardScaler()
cols_to_scale = ['Expansion_Index', 'Financial_Sentiment', 'Log_RD', 'ROE', 'Ln_Expansion_Count']
X_scaled = pd.DataFrame(scaler.fit_transform(model_df[cols_to_scale]), columns=cols_to_scale, index=model_df.index)
y_revenue = model_df['Log_Revenue']
y_rd = model_df['Log_RD']
y_cash = model_df['Log_CashFlow']

# --- 模型 1: 对照组验证 (Financial Sentiment -> Revenue) ---
X1 = sm.add_constant(X_scaled[['Financial_Sentiment', 'ROE']])
model1 = sm.OLS(y_revenue, X1).fit()
print("\n[Model 1] 对照组: 通用情感 -> 营收")
print(f"Sentiment P-value: {model1.pvalues['Financial_Sentiment']:.4f} (预期不显著)")

# --- 模型 2: 核心主效应 (Expansion Index -> Revenue) ---
X2 = sm.add_constant(X_scaled[['Expansion_Index', 'ROE']])
model2 = sm.OLS(y_revenue, X2).fit()
print("\n[Model 2] 主效应: 战略扩张 -> 营收")
print(f"Expansion P-value: {model2.pvalues['Expansion_Index']:.4f} (预期显著)")

# --- 模型 3: 机制前半段 (Expansion Index -> R&D) ---
X3 = sm.add_constant(X_scaled[['Expansion_Index', 'ROE']])
model3 = sm.OLS(y_rd, X3).fit()
print("\n[Model 3] 机制验证: 战略扩张 -> 研发投入")
print(f"Expansion -> R&D P-value: {model3.pvalues['Expansion_Index']:.4f} (预期显著)")

# --- 模型 4: 中介效应检验 (Expansion + R&D -> Revenue) ---
X4 = sm.add_constant(X_scaled[['Expansion_Index', 'Log_RD', 'ROE']])
model4 = sm.OLS(y_revenue, X4).fit()
print("\n[Model 4] 中介效应: 战略扩张 + 研发投入 -> 营收")
print(model4.summary().tables[1])

# ================= 5.5 稳健性检验 (Robustness Check) =================
print("\n🛡️ 5.5 运行稳健性检验...")

# --- Robustness 1: 替换被解释变量 (Y = Log_CashFlow) ---
# 逻辑：检验战略扩张是否驱动了真金白银的现金流
X_rob1 = sm.add_constant(X_scaled[['Expansion_Index', 'ROE']])
model_rob1 = sm.OLS(y_cash, X_rob1).fit()
print("\n[Robustness 1] 替换 Y: 战略扩张 -> 现金流 (Log_CashFlow)")
print(f"Expansion -> CashFlow P-value: {model_rob1.pvalues['Expansion_Index']:.4f}")

# --- Robustness 2: 替换解释变量 (X = Ln_Expansion_Count) ---
# 逻辑：检验单纯的扩张词频计数（取对数）是否依然显著，排除密度计算公式的干扰
X_rob2 = sm.add_constant(X_scaled[['Ln_Expansion_Count', 'ROE']])
model_rob2 = sm.OLS(y_revenue, X_rob2).fit()
print("\n[Robustness 2] 替换 X: 扩张词频对数 -> 营收")
print(f"Ln_Count -> Revenue P-value: {model_rob2.pvalues['Ln_Expansion_Count']:.4f}")

print("✅ 回归分析完成。")

# ================= 6. 可视化输出 =================
import seaborn as sns

# 数据插值用于绘图平滑
plot_df = df.copy()
num_cols = plot_df.select_dtypes(include=[np.number]).columns
plot_df[num_cols] = plot_df[num_cols].interpolate()

# 辅助函数：Z-Score 标准化
def z_score(series):
    return (series - series.mean()) / series.std()

# --- 图 1: LDA 战略演进河流图 ---
plt.figure(figsize=(12, 6))
plt.stackplot(plot_df['Year'], plot_df[topic_cols].T, labels=['Topic 0 (转型/多元)', 'Topic 1 (技术/爆发)', 'Topic 2 (代工/起步)'], alpha=0.85)
plt.legend(loc='upper left', title='Strategic Topics', bbox_to_anchor=(1, 1))
plt.title('比亚迪 2002-2024 战略重心演变 (LDA Topic Evolution)', fontsize=14, fontweight='bold')
plt.xlabel('年份')
plt.ylabel('主题权重')
plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(path_chart1)
print(f"📊 图表 1 已保存: {path_chart1}")
plt.show()

# --- 图 2: 战略-研发-营收 三维传导全景图 (增加公式) ---
fig, ax1 = plt.subplots(figsize=(13, 7))
viz_df = plot_df[plot_df['Year'] > 2005].copy()

# 产出层 (营收)
ax1.set_xlabel('年份', fontsize=12)
ax1.set_ylabel('营收规模 (标准化后)', color='green', fontsize=12, fontweight='bold')
ax1.fill_between(viz_df['Year'], z_score(viz_df['Log_Revenue']), color='green', alpha=0.15, label='营收规模 (产出)')
ax1.tick_params(axis='y', labelcolor='green')
ax1.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))

# 驱动层 (战略 & 研发)
ax2 = ax1.twinx()
ax2.set_ylabel('驱动力强度 (标准化后)', color='black', fontsize=12, fontweight='bold')
ax2.plot(viz_df['Year'], z_score(viz_df['Expansion_Index']), color='#d62728', linewidth=3, marker='o', label='战略扩张指数 (意图)')
ax2.plot(viz_df['Year'], z_score(np.log(viz_df['RDExpense_Billion'])), color='#1f77b4', linewidth=2.5, linestyle='--', marker='x', label='研发投入 (动作)')

lines, labels = ax2.get_legend_handles_labels()
patch = mpatches.Patch(color='green', alpha=0.15, label='营收规模 (产出)')
ax2.legend([patch] + lines, ['营收规模 (产出)'] + labels, loc='upper left')

# 【新增】公式标注 (修复 LaTeX 报错：将 R\&D 改为 RD)
formula_text = (
    r"$Strategic\ Index = \frac{N_{pos}-N_{neg}}{N_{total}} \times 1000$" + "\n" +
    r"$Revenue = \ln(Revenue)$" + "\n" +
    r"$RD = \ln(RD\ Expense)$"
)
plt.figtext(0.065, 0.72, formula_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.title('机制验证：战略意图(红) 转化为 研发投入(蓝)，最终驱动 营收增长(绿)', fontsize=15)
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(path_chart2)
print(f"📊 图表 2 已保存: {path_chart2}")
plt.show()

# --- 图 3: "言行一致" 散点回归图 (增加公式) ---
plt.figure(figsize=(10, 6))
sns.regplot(x=viz_df['Expansion_Index'], y=np.log(viz_df['RDExpense_Billion']),
            color="#d62728", scatter_kws={'s': 50}, line_kws={'linewidth': 2})
for i, row in viz_df.iterrows():
    if row['Year'] in [2009, 2015, 2019, 2021, 2023]:
        plt.text(row['Expansion_Index']+0.2, np.log(row['RDExpense_Billion']), str(int(row['Year'])), fontsize=9)

# 【新增】公式标注 (修复 LaTeX 报错)
plt.text(6.7,5.4,
         r"$Y = \ln(RD\ Investment)$" + "\n" + r"$X = Strategic\ Expansion\ Index$",
         fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

plt.title('路径检验：战略意图(言) 对 研发投入(行) 的强解释力', fontsize=14, fontweight='bold')
plt.xlabel('战略扩张指数 (Strategic Intent)', fontsize=12)
plt.ylabel('研发投入对数 (R&D Investment)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(path_chart3)
print(f"📊 图表 3 已保存: {path_chart3}")
plt.show()

# --- 图 4: 战略韧性指数 (Resilience Score) (增加公式) ---
# 韧性 = 扩张意图(Z) - 财务情感(Z)。含义：在市场情绪低迷时，依然保持高扩张意图，体现“韧性”。
df['Resilience_Score'] = z_score(df['Expansion_Index']) - z_score(df['Financial_Sentiment'])

plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['Resilience_Score'], label='Strategic Resilience (战略韧性)', color='purple', linewidth=3, marker='D')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
# 标注 2019 年
if 2019 in df['Year'].values:
    val_2019 = df[df['Year'] == 2019]['Resilience_Score'].values[0]
    plt.annotate('2019: 财务至暗，韧性最强', xy=(2019, val_2019), xytext=(2017, val_2019+1),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, fontweight='bold')

# 【新增】公式标注
plt.title('比亚迪“战略韧性”指数 (2002-2024)：逆境中的扩张定力', fontsize=14)
plt.figtext(0.05, 0.80, r"$Resilience = Z(Expansion) - Z(Sentiment)$", fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('年份')
plt.ylabel('韧性指数 (Resilience Score)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(path_chart4)
print(f"📊 图表 4 (韧性指数) 已保存: {path_chart4}")
plt.show()

# 保存最终表格
df.to_csv(path_final_csv, index=False, encoding='utf-8-sig')
print(f"💾 最终数据表已保存: {path_final_csv}")