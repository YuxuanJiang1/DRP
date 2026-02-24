#!/bin/bash

SESSION=math  # Existing tmux session name

# Step 1: Start training
tmux send-keys -t $SESSION "conda activate qwen" C-m
tmux send-keys -t $SESSION "echo '🚀 Step 1: Starting SFT training...'" C-m
tmux send-keys -t $SESSION "CUDA_VISIBLE_DEVICES=0 llamafactory-cli train /path/eval/qwen_lora_sft.yaml | tee sft_train.log" C-m

# Step 2 and 3: Execute after training completes
tmux send-keys -t $SESSION '
echo "🕒 Waiting for training to complete, then starting Step 2..." &&
conda deactivate &&
source activate /p/work2/yuxuanj1/conda_envs/qwen &&
cd /p/work2/yuxuanj1/reasoning/eval &&
echo "🔗 Step 2: Merging LoRA weights..." &&
python3 lora_merge.py | tee lora_merge.log &&
cd eval_sft &&
echo "🧪 Step 3: Running evaluation script..." &&
python3 sft_gsm8k.py | tee sft_eval.log &&
echo "✅ All steps completed successfully!"
' C-m

echo "🟢 All commands have been sent to tmux session [$SESSION]. You can view the process using:"
echo "    tmux attach -t $SESSION"
