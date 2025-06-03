#!/bin/bash

# ===========================
# DRP: Efficient Reasoning Pipeline
# Script: run_all.sh
# Purpose: Run training + LoRA merge + evaluation automatically via tmux
# ===========================

# 🟦 Customize these paths before running
PROJECT_DIR=$(pwd)                                # Automatically points to current project root
TRAIN_ENV_PATH="$PROJECT_DIR/../train_env/bin/activate"   # Update if you store env elsewhere
EVAL_ENV_PATH="$PROJECT_DIR/../eval_env/bin/activate"     # Update if you store env elsewhere
SESSION=math                                      # You may rename the tmux session if needed

# 🧠 Step 1: LoRA SFT training
tmux send-keys -t $SESSION "
source $TRAIN_ENV_PATH &&
echo '🚀 Step 1: Starting SFT training...' &&
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train $PROJECT_DIR/configs/qwen_lora_sft.yaml | tee $PROJECT_DIR/logs/sft_train.log
" C-m

# 🔁 Step 2 & 3: Merge weights + Evaluate
tmux send-keys -t $SESSION "
echo '🕒 Waiting for training to complete, then starting Step 2...' &&
source $EVAL_ENV_PATH &&
cd $PROJECT_DIR &&
echo '🔗 Step 2: Merging LoRA weights...' &&
python3 scripts/lora_merge.py | tee logs/lora_merge.log &&
echo '🧪 Step 3: Running evaluation script...' &&
python3 scripts/sft_gsm8k.py | tee logs/sft_eval.log &&
echo '✅ All steps completed successfully!'
" C-m

# 🖥️ Final instructions
echo "🟢 Commands sent to tmux session [$SESSION]."
echo "💡 To monitor progress, run: tmux attach -t $SESSION"
