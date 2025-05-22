## DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models

This is the official repository of the paper [DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models](https://arxiv.org/abs/2505.13975).

- If you find our work helpful and it has been of any assistance to you, we would greatly appreciate it if you could kindly cite it:
  
```
@misc{jiang2025drpdistilledreasoningpruning,
      title={DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models}, 
      author={Yuxuan Jiang and Dawei Li and Frank Ferraro},
      year={2025},
      eprint={2505.13975},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.13975}, 
}
```

## 🚀 Introduction

While Large Reasoning Models (LRMs) have demonstrated remarkable success in complex reasoning tasks through Long Chain-of-Thought (CoT) reasoning, their inference often involves excessively verbose reasoning traces, resulting in substantial inefficiency. To address this issue, we propose \textbf{Distilled Reasoning Pruning (DRP)}, a hybrid framework that combines inference-time pruning with tuning-based distillation---two widely used strategies for efficient reasoning. DRP employs a teacher model to perform skill-aware step decomposition and content pruning, and then distills the pruned reasoning paths into a student model, enabling it to reason both efficiently and accurately. Across a series of challenging mathematical reasoning datasets, we find models trained with DRP achieve substantial improvements in token efficiency without sacrificing accuracy. Specifically, DRP reduces the average token usage on GSM8K from 917 to 328 while improving accuracy from 91.7% to 94.1%, and achieves a 43% token reduction on AIME with no performance drop. Further analysis reveals that aligning the reasoning structure of training CoTs with the student’s reasoning capacity is critical for effective knowledge transfer and performance gains.

<div style="text-align: center;">
  <img src="./resources/overview.png" width="800" >
</div>

<div style="text-align: center;">
  <img src="./resources/method.png" width="1000" >
</div>

## 📄 Get Started

### 📝 Setup
- Firstly, install the required environment:
```
conda create -n pl python==3.10

conda activate pl

pip install -r requirements.txt

# important package
deepspeed=0.14.4
flash-attn=2.3.6
llamafactory=0.9.2.dev0
transformers=4.48.1
vllm=0.6.1.post1+cu118
```
- Next, get and fill all the required API. In this work, we use [GPT-4o](https://openai.com/index/gpt-4/), [Gemini-2.0-flash](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.0-flash) and [Chatgpt](https://openai.com/index/chatgpt/).
  
### 💻 Models

We use [R1-Distill-Qwen-7B]([https://huggingface.co/mistralai/Mistral-7B-v0.1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)) for our main experiment. Please first get the access of that model.

### 📥 Data

Coming soon.


## ⛳️ Run

To make the analysis convinient, we released all the judgment results in `analysis/alpacaEval_result/` and `analysis/arenaHard_result/`

### Main Experiment

- First run the following command to train the student models:
```
  bash training/train_sft.sh
```


<div style="text-align: center;">
  <img src="./resource/token capture.png" width="500">
</div>


<div style="text-align: center;">
  <img src="./resource/token shift.png" width="1000">
</div>

## Acknowledge

- This work borrows and forks the following repositories for training and evaluation: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main), and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
