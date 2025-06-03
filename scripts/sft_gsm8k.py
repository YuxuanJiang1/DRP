import os
import subprocess
import glob
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from lm_eval.tasks import get_task_dict

# ====== User Configuration ======
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Task to evaluate (GSM8K with chain-of-thought, zero-shot)
task_name = "gsm8k_cot_zeroshot"

# Path to merged model (after LoRA merge)
merged_model_path = os.path.join(PROJECT_DIR, "checkpoints/merged_model")

# Output directory for evaluation results
output_dir = os.path.join(PROJECT_DIR, "results", "gsm8k_cot_eval")
os.makedirs(output_dir, exist_ok=True)

# GPU settings
gpu_id = "0"
max_gen_tokens = "40000"  # Recommended: adjust to prevent long outputs
# ================================

# Load tokenizer for token analysis
tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)

# ====== Utility Functions ======

def get_latest_sample_file(output_dir, task_name):
    """Find the most recent sample output file for a given task."""
    pattern = os.path.join(output_dir, f"{task_name}_full.json", "**", f"samples_{task_name}_*.jsonl")
    sample_files = glob.glob(pattern, recursive=True)
    return max(sample_files, key=os.path.getmtime) if sample_files else None

def is_already_processed(task_name, output_dir):
    """Check if evaluation has already been completed by comparing sample count."""
    sample_file = get_latest_sample_file(output_dir, task_name)
    if not sample_file:
        return False
    try:
        task = get_task_dict([task_name])[task_name]
        docs = list(task.validation_docs() or task.test_docs())
        expected = len(docs)
        with open(sample_file, "r", encoding="utf-8", errors="ignore") as f:
            actual = len(f.readlines())
        return actual >= expected
    except Exception as e:
        print(f"⚠️ Error checking sample count for {task_name}: {e}")
        return False

def print_token_summary(task_name, output_dir):
    """Print token-level statistics of the generated responses."""
    print(f"\n=== Token Count Summary for {task_name} ===")
    sample_file = get_latest_sample_file(output_dir, task_name)
    if not sample_file:
        print("❌ No sample file found.")
        return
    try:
        total_tokens = 0
        max_tokens_seen = 0
        count = 0
        with open(sample_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc="Tokenizing responses"):
                data = json.loads(line)
                resps = data.get("resps", [])
                if isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list):
                    response_text = resps[0][0]
                    tokens = len(tokenizer.encode(response_text))
                    total_tokens += tokens
                    max_tokens_seen = max(max_tokens_seen, tokens)
                    count += 1
        if count > 0:
            print(f"📄 Total responses: {count}")
            print(f"🔢 Avg token count: {total_tokens / count:.2f}")
            print(f"📊 Max token count: {max_tokens_seen}")
            print(f"📈 Total tokens: {total_tokens}")
        else:
            print("⚠️ No valid responses found.")
    except Exception as e:
        print(f"⚠️ Error reading sample file: {e}")

# ====== Main Execution ======

print(f"\n=== Running {task_name} with merged model and vLLM ===")

if is_already_processed(task_name, output_dir):
    print(f"✅ Already processed {task_name}, skipping.")
else:
    output_path = os.path.join(output_dir, f"{task_name}_full.json")
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", (
            f"pretrained={merged_model_path},"
            "trust_remote_code=True,"
            "tensor_parallel_size=1,"
            "gpu_memory_utilization=0.6,"
            "dtype=half"
        ),
        "--tasks", task_name,
        "--num_fewshot", "0",
        "--output_path", output_path,
        "--batch_size", "1",
        "--gen_kwargs", f"temperature=0.0,top_p=1.0,max_gen_toks={max_gen_tokens}"
    ]

    subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id})
    print_token_summary(task_name, output_dir)
