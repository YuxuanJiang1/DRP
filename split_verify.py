import pandas as pd
import openai
import re
import json
import os

# ✅ Set API Key
openai.api_key = ''

# ✅ Path Configuration
input_csv = "R1-Qwen-7B/7B_output_on_train.csv"
intermediate_csv = "splitverify/gpt4o_output_with_steps_batch.csv"
final_csv = "splitverify/gpt4o_verified_compressed_batch.csv"

os.makedirs("batch_inputs", exist_ok=True)
os.makedirs("batch_outputs", exist_ok=True)

# ✅ Prompt Templates
step_segmentation_prompt = """You will be provided with a full reasoning path. 
Your task is to break it into a sequence of clear, non-overlapping steps, where each step corresponds to exactly one atomic mathematical skill (e.g., addition, subtraction, applying a formula, interpreting a quantity, simplifying, checking a condition).
Do not skip or rewrite anything. Cover every part of the original text — all explanations, calculations, and equations — step by step, without omission.
Use the format: Step n: {{segment of original text}}\nSkill: {{skill used}}. Only split and label steps. Don't skip anything, like the last sentence: the final answer is xx. skill: clarify answer.
Only segment and label the steps — do not solve or modify original content."""

compression_prompt = """You are an expert in mathematical reasoning compression.

Given a list of reasoning steps, each labeled with the skill used, your task is to evaluate whether and how each step should be simplified to improve efficiency without losing necessary logic.

You have four actions to choose from for each step:

1. KEEP: The step is necessary and already concise. Keep it unchanged.
2. DELETE: The step is unnecessary and should be removed entirely.
3. SINGLE-STEP COMPRESS: The step is necessary but verbose; rewrite it in a more concise way.
4. MULTI-STEP COMPRESS: The step can be merged with neighboring steps; write a combined, cleaner version.

For the last step, if it's to clarify the final answer, like 'the answer is xx', keep it.

After you’ve completed the step evaluations and transformations, synthesize them into a single, coherent reasoning path that maintains a fluent tone and matches the original speaker’s style. At the end, make sure there is a clarification of the answer starting with 'The answer is'.

Output the integrated compressed reasoning starting with ##."""

# ✅ Load Data
existing = pd.read_csv(intermediate_csv) if os.path.exists(intermediate_csv) else pd.DataFrame()
already_processed = len(existing)
print(f"🔍 Already processed {already_processed} records.")

df_full = pd.read_csv(input_csv)
df = df_full.iloc[already_processed:].reset_index(drop=True)
print(f"📥 Number of new samples to process: {len(df)}")

# ✅ Build batch input
def build_batch_input_file(df, output_jsonl_path, task_type):
    batch_data = []
    for idx, row in df.iterrows():
        real_idx = idx + already_processed
        if task_type == "step_segmentation":
            qwen_response = row["qwen_full_response"]
            match = re.search(r"<think>(.*?)</think>", qwen_response, re.DOTALL)
            reasoning_text = match.group(1).strip() if match else qwen_response
            sys_prompt = step_segmentation_prompt
            user_prompt = f"<think>\n{reasoning_text}\n</think>"
        elif task_type == "compression":
            sys_prompt = compression_prompt
            user_prompt = f"Question:\n{row['question']}\n\nReasoning Steps:\n{row['reasoning_steps']}"
        batch_data.append({
            "custom_id": str(real_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0,
                "max_tokens": 2048
            }
        })
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in batch_data:
            f.write(json.dumps(item) + "\n")

# ✅ Upload and create batch job
def upload_and_create_job(jsonl_path):
    file_id = openai.files.create(file=open(jsonl_path, "rb"), purpose="batch").id
    batch_id = openai.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions", completion_window="24h").id
    with open("saved_batch_id.txt", "w") as f:
        f.write(batch_id)
    print(f"🚀 Batch ID {batch_id} saved to saved_batch_id.txt")
    return batch_id

# ✅ Resume and download batch output to specified path
def resume_and_download(output_path):
    with open("saved_batch_id.txt") as f:
        batch_id = f.read().strip()
    status = openai.batches.retrieve(batch_id)
    print("⏳ Current status:", status.status)
    if status.status == "completed":
        output_id = status.output_file_id
        content = openai.files.content(output_id).text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Download completed and saved to: {output_path}")
    else:
        print("⌛ Not yet completed, try again later")

# ✅ Attach results to DataFrame
def attach_batch_results(df, output_path, col, is_compression=False):
    with open(output_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if is_compression:
        full = {int(i["custom_id"]): i["response"]["body"]["choices"][0]["message"]["content"] for i in data}
        compressed = {}
        for k, v in full.items():
            match = re.search(r"##\s*(.*)", v, re.DOTALL)
            compressed[k] = match.group(1).strip() if match else "[no ## result found]"
        df["verified_response"] = df.index.map(lambda i: full.get(i + already_processed))
        df["compressed_response"] = df.index.map(lambda i: compressed.get(i + already_processed))
    else:
        mapping = {int(i["custom_id"]): i["response"]["body"]["choices"][0]["message"]["content"] for i in data}
        df[col] = df.index.map(lambda i: mapping.get(i + already_processed))

# ✅ Merge and save results
def merge_and_save(df, output_csv):
    old = pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()
    merged = pd.concat([old, df], ignore_index=True)
    merged.to_csv(output_csv, index=False)
    print(f"💾 Merged results saved to: {output_csv}")

# # ✅ Main workflow
# # —— Step 1: Step Segmentation ——
# build_batch_input_file(df, "batch_inputs/step_segmentation_input.jsonl", "step_segmentation")
# upload_and_create_job("batch_inputs/step_segmentation_input.jsonl")
#
# —— Step 2: Download step segmentation results ——
resume_and_download("batch_outputs/step_segmentation_output.json")
attach_batch_results(df, "batch_outputs/step_segmentation_output.json", "reasoning_steps")
merge_and_save(df, intermediate_csv)

# —— Step 3: Construct compression tasks ——
if len(df) == 0:
    print("✅ No new data to compress, skipping upload.")
else:
    build_batch_input_file(df, "batch_inputs/compression_input.jsonl", "compression")
    upload_and_create_job("batch_inputs/compression_input.jsonl")

# —— Step 4: Download compression results and save ——
resume_and_download("batch_outputs/compression_output.json")
attach_batch_results(df, "batch_outputs/compression_output.json", "compressed_response", is_compression=True)
merge_and_save(df, final_csv)

print("🎉 Full pipeline completed!")
