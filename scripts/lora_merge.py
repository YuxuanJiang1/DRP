import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ====== User Configuration ======
# Set these paths relative to the project root

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to the base model (before fine-tuning)
base_model_path = os.path.join(PROJECT_DIR, "checkpoints/base_model")

# Path to the LoRA adapter directory (after fine-tuning)
adapter_path = os.path.join(PROJECT_DIR, "checkpoints/lora_adapter")

# Output path to save the merged model
output_path = os.path.join(PROJECT_DIR, "checkpoints/merged_model")
# ===============================

os.makedirs(output_path, exist_ok=True)

print("🔄 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype="auto"
)

print("🔌 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("🧬 Merging adapter into base model...")
model = model.merge_and_unload()

print(f"💾 Saving merged model to: {output_path}")
model.save_pretrained(output_path)

# Optional: save tokenizer as well (recommended to keep versions in sync)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print("✅ Merge complete. Model is ready to be used with vLLM.")

