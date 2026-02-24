import os
import subprocess
import glob
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from lm_eval.tasks import get_task_dict

# 模型路径
model_path = "/p/work2/yuxuanj1/hf_models/DeepSeek-R1-Distill-Qwen-1.5B"

# 数据集列表
# datasets = ["hendrycks_math_500", "AIME", "AMC"]
datasets = ["gsm8k_cot_zeroshot"]
# 输出目录
output_dir = "/p/work2/yuxuanj1/reasoning/baselines/1.5B"
os.makedirs(output_dir, exist_ok=True)

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def get_latest_sample_file(output_dir, task_name):
    pattern = os.path.join(output_dir, f"{task_name}_full.json", "**", f"samples_{task_name}_*.jsonl")
    sample_files = glob.glob(pattern, recursive=True)
    return max(sample_files, key=os.path.getmtime) if sample_files else None


def is_already_processed(task_name, output_dir):
    sample_file = get_latest_sample_file(output_dir, task_name)
    if not sample_file:
        return False
    try:
        task = get_task_dict([task_name])[task_name]
        docs = list(task.validation_docs() or task.test_docs())
        expected = len(docs)

        with open(sample_file, "r") as f:
            actual = len(f.readlines())
        return actual >= expected
    except Exception as e:
        print(f"⚠️ Error checking sample count for {task_name}: {e}")
        return False


def print_token_summary(task_name, output_dir):
    print(f"\n=== Token Count Summary for {task_name} ===")
    sample_file = get_latest_sample_file(output_dir, task_name)

    if not sample_file:
        print("❌ No sample file found.")
        return

    try:
        total_tokens = 0
        max_tokens = 0
        count = 0

        with open(sample_file, "r") as f:
            for line in tqdm(f, desc="Tokenizing responses"):
                data = json.loads(line)
                resps = data.get("resps", [])
                if isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list):
                    response_text = resps[0][0]
                    tokens = len(tokenizer.encode(response_text))
                    total_tokens += tokens
                    max_tokens = max(max_tokens, tokens)
                    count += 1

        if count > 0:
            print(f"📄 Total responses: {count}")
            print(f"🔢 Average token count: {total_tokens / count:.2f}")
            print(f"📊 Max token count: {max_tokens}")
            print(f"📈 Total tokens: {total_tokens}")
        else:
            print("⚠️ No valid responses found.")

    except Exception as e:
        print(f"⚠️ Error reading sample file: {e}")


# ========= 主运行逻辑 =========
for task in datasets:
    print(f"\n=== Running {task} ===")

    if is_already_processed(task, output_dir):
        print(f"✅ Already processed {task}, skipping.")
        continue

    max_tokens = "80000" if task == "hendrycks_math_500" else "130000"
    output_path = os.path.join(output_dir, f"{task}_full.json")


    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", (
            "pretrained=/p/work2/yuxuanj1/hf_models/DeepSeek-R1-Distill-Qwen-1.5B,"
            "trust_remote_code=True,"
            # "max_model_len=8000,"

            "tensor_parallel_size=1,"
            "gpu_memory_utilization=0.6,"
            "dtype=half"
        ),
        "--tasks", task,
        "--num_fewshot", "0",
        "--output_path", output_path,
        "--batch_size", "8",
        "--gen_kwargs", "temperature=0.0,top_p=1.0,max_gen_toks=40000"
    ]
    subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    print_token_summary(task, output_dir)
