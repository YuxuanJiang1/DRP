import pandas as pd
import re
import openai
from tqdm import tqdm
import tiktoken

# 设置 OpenAI API Key
openai.api_key = ('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')

# 读取原始 GPT-4o 输出数据
df = pd.read_csv("R1-Qwen-7B/7B_output_on_train.csv").head(100)

# ---------- 第一阶段：提取 <think> 推理路径并分步 ---------- #

step_segmentation_prompt = (
    """You will be provided with a full reasoning path. 
Your task is to break it into a sequence of clear, non-overlapping steps, where each step corresponds to exactly one atomic mathematical skill (e.g., addition, subtraction, applying a formula, interpreting a quantity, simplifying, checking a condition).
Do not skip or rewrite anything. Cover every part of the original text — all explanations, calculations, and equations — step by step, without omission.
Use the format: Step n: {segment of original text}\nSkill: {skill used}. Only split and label steps. Don't skip anything, like the last sentence: the final answer is xx, you can write down:step n:the answer is xx. skill: clarify answer.
Only segment and label the steps — do not solve or modify original content."""
)

def extract_and_split_think(response):
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    reasoning_text = match.group(1).strip() if match else response

    messages = [
        {"role": "system", "content": step_segmentation_prompt},
        {"role": "user", "content": f"<think>\n{reasoning_text}\n</think>"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Step extraction error:", e)
        return "[error]"

print("🚀 第一步：提取 reasoning steps 中...")
df["reasoning_steps"] = [extract_and_split_think(resp) for resp in tqdm(df["qwen_full_response"])]

# 保存步骤拆分中间结果
df.to_csv("baselines/gpt4o_output_with_steps.csv", index=False, encoding="utf-8")

# ---------- 第二阶段：压缩步骤 + 生成流畅推理 ---------- #

compression_prompt = (
    "You are an expert in mathematical reasoning compression.\n\n"
    "Given a list of reasoning steps, each labeled with the skill used, your task is to evaluate whether and how each step should be simplified to improve efficiency without losing necessary logic.\n\n"
    "You have four actions to choose from for each step:\n\n"
    "1. KEEP: The step is necessary and already concise. Keep it unchanged.\n"
    "2. DELETE: The step is unnecessary and should be removed entirely.\n"
    "3. SINGLE-STEP COMPRESS: The step is necessary but verbose; rewrite it in a more concise way.\n"
    "4. MULTI-STEP COMPRESS: The step can be merged with neighboring steps; write a combined, cleaner version.\n\n"
    "For the last step, if it's to clarify the final answer,like 'the answer is xx', keep it.\n"
    "After you’ve completed the step evaluations and transformations, synthesize them into a single, coherent reasoning path that maintains a fluent tone and matches the original speaker’s style. At the end ,please make sure there is a clarification of the answer, with 'The answer is'.\n\n"
    "Output the integrated compressed reasoning starting with ##."
)

def compress_and_verify(question, reasoning_steps_text):
    user_input = f"Question:\n{question}\n\nReasoning Steps:\n{reasoning_steps_text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": compression_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            max_tokens=1024
        )
        full_output = response.choices[0].message.content.strip()
        match = re.search(r"##\s*(.*)", full_output, re.DOTALL)
        compressed_response = match.group(1).strip() if match else "[no ## result found]"
        return full_output, compressed_response
    except Exception as e:
        print("❌ Compression error:", e)
        return "[error]", "[error]"

print("🚀 第二步：验证并压缩 reasoning steps...")
verified_outputs = []
compressed_outputs = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    full, compressed = compress_and_verify(row["question"], row["reasoning_steps"])
    verified_outputs.append(full)
    compressed_outputs.append(compressed)

df["verified_response"] = verified_outputs
df["compressed_response"] = compressed_outputs

# 保存最终结果
output_path = "baselines/gpt4o_verified_compressed.csv"
df[["question", "verified_response", "compressed_response"]].to_csv(output_path, index=False, encoding="utf-8")


def analyze_token_usage(df, output_csv_path):
    print("\n📊 正在分析 token 数量...")

    # 初始化 tokenizer
    enc = tiktoken.encoding_for_model("gpt-4o")

    # 定义计数函数
    def count_tokens(text):
        return len(enc.encode(text)) if isinstance(text, str) else 0

    # 逐行计算
    df["verified_token_count"] = df["verified_response"].apply(count_tokens)
    df["compressed_token_count"] = df["compressed_response"].apply(count_tokens)

    # 汇总信息
    total_verified = df["verified_token_count"].sum()
    total_compressed = df["compressed_token_count"].sum()
    avg_verified = df["verified_token_count"].mean()
    avg_compressed = df["compressed_token_count"].mean()

    # 打印结果
    print("🧮 Token统计结果：")
    print(f"✅ 原始 verified_response 总token数: {total_verified}，平均每条: {avg_verified:.2f}")
    print(f"✅ 压缩 compressed_response 总token数: {total_compressed}，平均每条: {avg_compressed:.2f}")
    print(f"🎯 节省比例: {(1 - total_compressed / total_verified) * 100:.2f}%")

    # 保存包含 token 数的新 CSV
    output_with_tokens = output_csv_path.replace(".csv", "_token_count.csv")
    df.to_csv(output_with_tokens, index=False, encoding="utf-8")
    print(f"📁 保存含 token 统计的文件至：{output_with_tokens}")

# 运行 token 分析（添加在主流程末尾）
analyze_token_usage(df, output_path)

print(f"✅ 所有处理完成，结果保存至：{output_path}")
