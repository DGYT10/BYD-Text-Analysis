import pandas as pd
import json
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# ================= 配置区域 =================
# 1. 动态获取路径
# 获取 train_bert.py 所在的文件夹 (即 .../da/final)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接路径：从当前文件夹 -> 进入 data -> 进入 FinEntCN -> 找到文件
# 你的文件结构是：final/train_bert.py 和 final/data/ 是兄弟关系
TRAIN_FILE = os.path.join(current_dir, 'data', 'FinEntCN', 'train_org.json')

# 如果没有单独的测试集文件，留空 None
TEST_FILE = os.path.join(current_dir, 'data', 'FinEntCN', 'test_org.json')
# 如果你确定文件夹里有 test_org.json，就取消下面这行的注释
# TEST_FILE = os.path.join(current_dir, 'data', 'FinEntCN', 'test_org.json')

# 2. 输出路径
OUTPUT_DIR = os.path.join(current_dir, 'data', 'my_financial_bert')

# 3. 标签映射 (FinEntCN 的标签是中文，需要转数字)
# 2=Positive, 1=Neutral, 0=Negative
label_map = {
    '正面': 2,
    '中性': 1,
    '负面': 0
}

# 打印路径
print(f"📂 脚本位置: {current_dir}")
print(f"📂 训练数据: {TRAIN_FILE}")
# =================================================================


def load_finentcn_data(file_path):
    """
    专门解析 FinEntCN 的特殊格式
    结构: list of dict
    关键字段: 'content' (文本), 'output' (包含 tag 的 JSON 字符串)
    """
    if not file_path or not os.path.exists(file_path):
        print(f"⚠️ 文件不存在或未指定: {file_path}")
        return None

    print(f"📖 正在读取: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []

    for item in data:
        try:
            # 1. 获取文本
            text = item.get('content', '')

            # 2. 获取标签 (这是难点)
            # 'output' 是一个字符串: "{\"tag\": \"正面\", ...}"
            # 需要用 json.loads 再解一层
            output_str = item.get('output', '{}')
            output_dict = json.loads(output_str)

            tag = output_dict.get('tag')  # 拿到 "正面" 或 "负面"

            # 3. 映射为数字
            if tag in label_map:
                label_id = label_map[tag]
                processed_data.append({'text': text, 'label': label_id})

        except Exception as e:
            # 某些行可能有格式错误，跳过即可
            continue

    df = pd.DataFrame(processed_data)
    print(f"✅ 成功解析 {len(df)} 条有效数据")
    return df


def train():
    # 1. 加载数据
    df_train = load_finentcn_data(TRAIN_FILE)

    # 尝试加载测试集，如果没有，就从训练集切分
    df_test = load_finentcn_data(TEST_FILE)

    if df_test is None or len(df_test) == 0:
        print("✂️ 未找到测试集文件，正在从训练集中切分 20% 作为测试集...")
        df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)

    print(f"📊 最终数据集: 训练集 {len(df_train)} 条 | 测试集 {len(df_test)} 条")
    print(f"🔍 样本预览: {df_train.iloc[0].to_dict()}")

    # 2. 转为 HuggingFace Dataset
    ds_train = Dataset.from_pandas(df_train)
    ds_test = Dataset.from_pandas(df_test)

    # 3. 加载 Tokenizer (使用中文 BERT 基座)
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("🔄 正在进行分词处理 (Tokenizing)...")
    tokenized_train = ds_train.map(tokenize_function, batched=True)
    tokenized_test = ds_test.map(tokenize_function, batched=True)

    # 4. 加载模型
    print("🤖 加载 BERT 模型...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 5. 训练参数 (针对新版 Transformers 和 M1 优化)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # 训练 3 轮
        per_device_train_batch_size=16,  # 批次大小
        per_device_eval_batch_size=16,

        # 【修改点 1】改名：evaluation_strategy -> eval_strategy
        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=2e-5,

        # 【修改点 2】删除 use_mps_device=True
        # 新版 transformers 会自动检测 M1 芯片 (MPS)，
        # 显式写这个参数反而会报错或警告
        # use_mps_device=True,  <-- 删除这一行

        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("🔥 开始微调 (Fine-tuning)... 请耐心等待")
    trainer.train()

    print(f"💾 模型已保存至: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)


if __name__ == '__main__':
    train()