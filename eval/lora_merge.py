import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ====== 用户设置区域 ======
base_model_path = "/p/work2/yuxuanj1/hf_models/DeepSeek-R1-Distill-Qwen-7B"
adapter_path = "/p/work2/yuxuanj1/ftmodels/qwen-7b-limo-3ep/lora-sft"
output_path = "/p/work2/yuxuanj1/merged_models/qwen-7b-limo-3ep"
# ==========================

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

print("💾 Saving merged model to:", output_path)
model.save_pretrained(output_path)

# 可选：也保存 tokenizer（建议一起用）
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print("✅ Merge complete. Model is ready to be used with vLLM.")
