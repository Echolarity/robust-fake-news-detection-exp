import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
torch.cuda.empty_cache()
# 模型名称（Qwen-7B对话版，效果最好）
model_id = "qwen/Qwen-7B-Chat"
# 自动下载模型（首次下载，后续直接加载本地）
# model_dir = snapshot_download(model_id)
# 改为使用已下载的完整模型路径
model_dir = "/root/.cache/modelscope/hub/models/qwen/Qwen-7B-Chat"
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True  # Qwen 模型必加
)

# 🔥 关键：3090 专属加载方式（FP16半精度，无量化，速度/效果拉满）
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # 半精度（3090完美支持）
    device_map="auto",           # 自动分配显卡（3090单卡最优）
    trust_remote_code=True,
    low_cpu_mem_usage=True       # 节省CPU内存
)

# 评估模式（加速推理）
model = model.eval()


def chat(query):
    # Qwen 官方对话模板（保证输出格式正确）
    torch.cuda.empty_cache()
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    try:
        # 推理（3090 秒级响应）
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,    # 3090 直接拉满最大生成长度
                temperature=0.7,        # 平衡严谨/创意
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1  # 防止重复输出
            )
        
        # 解析回答
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("assistant\n")[-1]
        return answer
    except Exception as e:
        print(f"ERROR!!!:{str(e)},原文本是{query}")
        return ""

# ===================== 启动对话 =====================
if __name__ == "__main__":
    print("✅ Qwen-7B 3090专属版启动成功！输入 退出 结束对话")
    while True:
        user_input = input("你：")
        if user_input in ["exit", "quit", "退出"]:
            print("👋 对话结束")
            break
        answer = chat(user_input)
        print(f"Qwen-7B：{answer}\n")