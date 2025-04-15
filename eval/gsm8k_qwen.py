import pandas as pd
import re
from tqdm import tqdm
import dashscope
import tiktoken

# 使用 tiktoken 模拟 GPT-4o tokenizer（估算用）
enc = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    return len(enc.encode(text)) if isinstance(text, str) else 0

# 数据路径
file_path = '../datasets/GSM8K/train.csv'
df = pd.read_csv(file_path).head(10)

# 答案提取
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer_hf(answer_text):
    match = ANS_RE.search(answer_text)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        try:
            return str(eval(match_str))
        except:
            return INVALID_ANS
    return INVALID_ANS

def extract_answer_model_output(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return str(eval(last_number))
    except:
        return INVALID_ANS

# Prompt
prompt = (
    "Your task is to answer the question below. Give step by step reasoning before you answer, "
    "and when you’re ready to answer, please use the format 'The answer is ####'. "
    "Here is an example for the answer format: The answer is #### 50"
)

# QWQ-32B 推理函数，返回两个部分
def query_qwq_dashscope(question):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]
    try:
        response = dashscope.Generation.call(
            api_key="",
            model="qwq-32b",
            messages=messages,
            stream=True
        )

        reasoning_content = ""
        answer_content = ""

        for chunk in response:
            msg = chunk.output.choices[0].message
            reasoning_content += msg.reasoning_content or ""
            answer_content += msg.content or ""

        return reasoning_content.strip(), answer_content.strip()
    except Exception as e:
        print("❌ DashScope 调用失败:", e)
        return "[error]", "[error]"

# 结果统计与收集
results = []
errors = []
correct_count = 0

reasoning_tokens = []
answer_tokens = []

print("🚀 开始调用 QWQ-32B 推理...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row["question"]
    reference_answer = extract_answer_hf(row["answer"])

    reasoning, answer = query_qwq_dashscope(question)
    full_response = reasoning + "\n" + answer
    predicted_answer = extract_answer_model_output(full_response)

    is_correct = predicted_answer == reference_answer
    if is_correct:
        correct_count += 1
    else:
        errors.append(f"Question: {question}\n")
        errors.append(f"QWQ Reasoning:\n{reasoning}\n")
        errors.append(f"QWQ Answer:\n{answer}\n")
        errors.append(f"Correct Answer: {reference_answer}\n")
        errors.append("=" * 50 + "\n")

    reasoning_tokens.append(count_tokens(reasoning))
    answer_tokens.append(count_tokens(answer))

    results.append({
        "question": question,
        "reference_answer": reference_answer,
        "qwen_answer": predicted_answer,  # 为兼容格式
        "is_correct": is_correct,
        "reasoning_content": reasoning,
        "answer_content": answer,
        "reasoning_token_count": count_tokens(reasoning),
        "answer_token_count": count_tokens(answer),
    })

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv("../baselines/qwq32B_train.csv", index=False, encoding="utf-8")

# 保存错误样本
if errors:
    with open("../baselines/qwq32B_train_error.txt", "w", encoding="utf-8") as f:
        f.writelines(errors)

# 统计与报告
accuracy = correct_count / len(df) * 100
avg_reasoning_tokens = sum(reasoning_tokens) / len(reasoning_tokens)
avg_answer_tokens = sum(answer_tokens+reasoning_tokens) / len(answer_tokens)

print(f"✅ QWQ-32B Accuracy: {accuracy:.2f}%")
print(f"🧮 平均 Reasoning Token 数: {avg_reasoning_tokens:.2f}")
print(f"🧮 平均 总Answer Token 数: {avg_answer_tokens:.2f}")
print("📁 结果已保存到 ../baselines/qwq32B_train.csv")
