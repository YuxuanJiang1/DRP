import os
import re
import pandas as pd
import tiktoken
import dashscope
from tqdm import tqdm

# ========== 配置 ==========
api_key = ""
dashscope.api_key = api_key

input_path = "baselines/qwq32B_train.csv"      # 需包含 reasoning_content 和 answer_content 字段
output_path = "qwq_output/qwq32b_final_combined.csv"
os.makedirs("qwq_output", exist_ok=True)

# ========== Prompt 1：step segmentation ==========
step_segmentation_prompt = (
    """Do not solve the problem. You will be provided with a full reasoning path. 
Your task is to break it into a sequence of clear, non-overlapping steps, 
where each step corresponds to exactly one atomic mathematical skill 
(e.g., addition, subtraction, applying a formula, interpreting a quantity, simplifying, checking a condition).

Do not skip or rewrite anything. Cover every part of the original text — 
all explanations, calculations, and equations — step by step, without omission.

Use the format: 
Step n: {segment of original text} 
Skill: {skill used}

Only split and label the steps — do not solve or modify original content."""
)

# ========== Prompt 2：compression ==========
compression_prompt = (
    """You are an expert in mathematical reasoning compression.

Given a list of reasoning steps, each labeled with the skill used, your task is to evaluate whether and how each step should be simplified to improve efficiency without losing necessary logic.

You have four actions to choose from for each step:

1. KEEP: The step is necessary and already concise. Keep it unchanged.
2. DELETE: The step is unnecessary and should be removed entirely.
3. SINGLE-STEP COMPRESS: The step is necessary but verbose; rewrite it in a more concise way.
4. MULTI-STEP COMPRESS: The step can be merged with neighboring steps; write a combined, cleaner version.

After you’ve completed the step evaluations and transformations, synthesize them into a single, coherent reasoning path that maintains a fluent tone and matches the original speaker’s style.

Output example:
step 1. DELETE: The step is unnecessary as it doesn't provide any new information.
step 2. SINGLE-STEP COMPRESS: James writes 4 letters per week (2 friends x 2 letters each).
step3. KEEP: This step is concise and provides necessary information.
step 4. MULTI-STEP COMPRESS: Multiply the weekly pages by the number of weeks in a year to find the total pages: 12 pages per week x 52 weeks = 624 pages.

final output: James writes 4 letters per week (2 friends x 2 letters each), and since each letter is 3 pages, he writes 12 pages per week (4 letters x 3 pages). There are 52 weeks in a year, so multiplying the weekly pages by the number of weeks gives us 624 pages in a year. The answer is 624 pages."""
)

# ========== 通用工具 ==========
enc = tiktoken.encoding_for_model("gpt-4o")
def count_tokens(text): return len(enc.encode(text)) if isinstance(text, str) else 0

def extract_final_output(text):
    if not isinstance(text, str): return ""
    match = re.search(r"(?:\*\*)?final output:?[\s\n]*(.*)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()

# ========== 调用 step segmentation ==========
def extract_and_split_think(reasoning_text):
    try:
        stream = dashscope.Generation.call(
            model="qwq-32b",
            messages=[
                {"role": "system", "content": step_segmentation_prompt},
                {"role": "user", "content": reasoning_text}
            ],
            stream=True,
        )
        reasoning_output, summary_output = "", ""
        for chunk in stream:
            msg = chunk.output.choices[0].message
            reasoning_output += msg.reasoning_content or ""
            summary_output += msg.content or ""
        return reasoning_output.strip(), summary_output.strip()
    except Exception as e:
        print("❌ Step segmentation error:", e)
        return "[error]", "[error]"

# ========== 调用压缩模块 ==========
def compress_with_actions(question, reasoning_steps_text):
    user_input = f"Question:\n{question}\n\nReasoning Steps:\n{reasoning_steps_text}"
    try:
        stream = dashscope.Generation.call(
            model="qwq-32b",
            messages=[
                {"role": "system", "content": compression_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=True,
        )
        reasoning_log = ""
        final_summary = ""
        for chunk in stream:
            msg = chunk.output.choices[0].message
            reasoning_log += msg.reasoning_content or ""
            final_summary += msg.content or ""
        return reasoning_log.strip(), final_summary.strip()
    except Exception as e:
        print("❌ Compression error:", e)
        return "[error]", "[error]"

# ========== 主执行流程 ==========
def run_full_pipeline(input_path, output_path, max_samples=None):
    df = pd.read_csv(input_path)
    if max_samples: df = df.head(max_samples)

    reasoning_steps, model_summaries = [], []
    compression_logs, compressed_responses, final_outputs = [], [], []
    original_outputs, original_token_counts, compressed_token_counts = [], [], []

    print("🚀 Step 1: 分步骤提取中...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        r_content, a_content = row["reasoning_content"], row["answer_content"]
        steps, summary = extract_and_split_think(r_content)
        reasoning_steps.append(steps)
        model_summaries.append(summary)

        # 原始输出拼接
        original_output = f"{r_content}\n{a_content}".strip()
        original_outputs.append(original_output)
        original_token_counts.append(count_tokens(original_output))

    df["reasoning_steps"] = reasoning_steps
    df["model_summary"] = model_summaries
    df["original_total_output"] = original_outputs
    df["original_token_count"] = original_token_counts

    print("🚀 Step 2: 压缩与 final output 提取中...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        log, compressed = compress_with_actions(row["question"], row["reasoning_steps"])
        compression_logs.append(log)
        compressed_responses.append(compressed)
        final_output = extract_final_output(compressed)
        final_outputs.append(final_output)
        compressed_token_counts.append(count_tokens(final_output))

    df["compression_log"] = compression_logs
    df["compressed_response"] = compressed_responses
    df["final_output_only"] = final_outputs
    df["compressed_token_count"] = compressed_token_counts

    avg_original = df["original_token_count"].mean()
    avg_compressed = df["compressed_token_count"].mean()

    print("\n🧮 Token 统计：")
    print(f"📌 原始输出 平均 token 数: {avg_original:.2f}")
    print(f"📌 压缩后 final output 平均 token 数: {avg_compressed:.2f}")
    print(f"🎯 节省比例: {(1 - avg_compressed / avg_original) * 100:.2f}%")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ 全流程已完成，结果已保存至：{output_path}")

# ========== 执行入口 ==========
if __name__ == "__main__":
    run_full_pipeline(input_path, output_path, max_samples=1)  # 设置为 None 跑全量
