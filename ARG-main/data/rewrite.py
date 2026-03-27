from load_news_json import load_json
from load_Qwen import chat
from data_process import save_json_linebyline
from tqdm import tqdm

if __name__ == "__main__":
    # DATA_PATH = "./convertedData.json"
    DATA_PATH = "./error_write_data.json"
    data, _, _, _, _ = load_json(DATA_PATH)
    CORRECT_PATH="1rewrite_data.json"
    ERROR_PATH="1error_write_data.json"
    
    ERROR = []
    CORRECT = []
    total_num=len(data)
    for item in tqdm(data,total=total_num,desc="重写进度",colour="blue"):
        content = item["content"]
        Prompt = f"""
请在不改变任何新闻类别标签的前提下，对给定的输入文本进行风格改写。目标是：
1. 忠实于新闻原意，不扭曲关键事实；
2. 将文本风格调整为类似微博、社交媒体的口语化表达，可酌情使用网络用语、语气词、缩写或口语句式，但保持信息完整；
3. 适当调整词汇、句法、表达结构，使改写后的新闻在内容含义不变的条件下，语言特征与原数据集形成差异；
4. 仅输出改写后的完整文本，不添加任何额外说明、标签或提示语。
输入文本为：{content}
        """
        try:
            answer = chat(Prompt)
            if answer.strip() == "":
                ERROR.append(item)
                continue
            item["content"] = answer.strip()
            CORRECT.append(item)
        except Exception as e:
            print(f"WRONG!!!处理失败：{content[:30]}...，错误信息：{str(e)}")
            ERROR.append(item)
            
    print(f"✅ 成功重写：{len(CORRECT)} 条")
    print(f"❌ 处理失败：{len(ERROR)} 条")
    save_json_linebyline(CORRECT_PATH,CORRECT)
    save_json_linebyline(ERROR_PATH,ERROR)
