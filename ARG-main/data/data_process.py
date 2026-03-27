from load_news_json import load_json, json
import random
from datetime import datetime

SEED = 42  # 随机种子
MIX_OUTPUT = "mixData.json"
CONVERTED_OUTPUT = "rewrite_data.json"
TRAIN_PATH = "rewrite/train.json"
VAL_PATH = "rewrite/val.json"
TEST_PATH = "rewrite/test.json"
fake_path = "/root/sunyuqi/fakenews/ARG-main/data/fake_release_all.json"
real_path = "/root/sunyuqi/fakenews/ARG-main/data/real_release_all.json"

data_real, real_r, count_real, real_rc, real_fc = load_json(real_path)
data_fake, fake_r, count_fake, fake_rc, fake_fc = load_json(fake_path)

# print("=" * 40)
# print("真实新闻统计")
# print("=" * 40)
# print(f"总数: {count_real}")
# print(f"真实率: {real_r}")
# print(f"真实数量: {real_rc}")
# print(f"虚假数量: {real_fc}")
# print()

# print("=" * 40)
# print("虚假新闻统计")
# print("=" * 40)
# print(f"总数: {count_fake}")
# print(f"真实率: {fake_r}")
# print(f"真实数量: {fake_rc}")
# print(f"虚假数量: {fake_fc}")
# print()
data = data_fake + data_real
count = count_fake + count_real
rate = (real_rc + fake_rc) / count


# print("=" * 40)
# print("新闻统计")
# print("=" * 40)
# print(f"总数: {count}")
# print(f"真实率: {rate}")
# print(f"真实数量: {real_rc+fake_rc}")
# print(f"虚假数量: {real_fc+fake_fc}")
# print()
def save_json_linebyline(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
def save_json_original(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def shuffle_save():
    random.seed(SEED)
    random.shuffle(data)
    save_json_linebyline(MIX_OUTPUT, data)
def convert_timestamp(ts):
    """毫秒级时间戳转可读格式（YYYY-MM-DD HH:MM:SS）"""
    try:
        # 确保ts是字符串或数字类型
        if isinstance(ts, str):
            ts = float(ts) if ts else 0
        else:
            ts = float(ts) if ts else 0

        ts_int = int(ts) // 1000
        return datetime.fromtimestamp(ts_int).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error converting timestamp {ts}: {e}")
        return ""
def convert_format(mfile_path):
    converted_data = []
    with open(mfile_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue
            try:
                raw_item = json.loads(line)
                converted_item = {
                    "content": raw_item.get("content", ""),  # 帖子内容
                    "label": raw_item.get("label"),  # 0=真，1=假（保持原始值）
                    "time": convert_timestamp(
                        raw_item.get("timestamp", "")
                    ),  # 发布时间
                    "source_id": raw_item.get("id", ""),  # 帖子唯一ID
                    # LLM相关字段（原始数据无，填充默认值，后续可替换为真实LLM输出）
                    "td_rationale": "",  # 文本描述视角LLM理由
                    "td_pred": "other",  # td_rationale预测结果（other=2）
                    "td_acc": 0,  # td_pred是否正确（0=错误）
                    "cs_rationale": "",  # 常识视角LLM理由
                    "cs_pred": "other",  # cs_rationale预测结果（other=2）
                    "cs_acc": 0,  # cs_pred是否正确（0=错误）
                    "split": "",  # 分割标签（第三步填充train/val/test）
                }
                converted_data.append(converted_item)
            except Exception as e:
                print(f"Error: {e}")
                continue
    # with open(CONVERTED_OUTPUT,'w',encoding='utf-8') as f:
    #     json.dump(converted_data,f,ensure_ascii=False,indent=4)
    save_json_linebyline(CONVERTED_OUTPUT, converted_data)
def split_dataset(train_ratio=0.8, val_ratio=0.1, seed=SEED):
    data, _, _, _, _ = load_json(CONVERTED_OUTPUT)
    lens = len(data)
    random.seed(seed)
    random.shuffle(data)

    train_num = int(lens * train_ratio)
    val_num = int(lens * val_ratio)
    # test_num=lens-train_num-val_num

    train_data = data[:train_num]
    val_data = data[train_num : train_num + val_num]
    test_data = data[train_num + val_num :]

    # print(type(train_data))
    # print(type(data))
    # print(data[0])
    for item in train_data:
        item["split"] = "train"

    for item in val_data:
        item["split"] = "valid"
    for item in test_data:
        item["split"] = "test"
    save_json_original(TRAIN_PATH, train_data)
    save_json_original(VAL_PATH, val_data)
    save_json_original(TEST_PATH, test_data)
if __name__ == "__main__":
    # shuffle_save()
    # convert_format(MIX_OUTPUT)
    split_dataset()
    print()
