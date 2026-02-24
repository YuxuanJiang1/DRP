import pandas as pd
import re
from tqdm import tqdm
import openai
import tiktoken


# Set OpenAI API Key (recommended to store in environment variables)
openai.api_key = ('')

# Use the tokenizer corresponding to GPT-4o
enc = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    return len(enc.encode(text))

# Load dataset
# file_path = "../datasets/GSM8K/test.csv"
file_path='datasets/GSM8K/train.csv'
df = pd.read_csv(file_path).head(2)

# More robust answer extraction regex and functions
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
prompt = "Your task is to answer the question below. Give step by step reasoning before you answer, and when you’re ready to answer, please use the format 'The answer is ####'. Here is an example for the answer format: The answer is #### 50"

# Use GPT-4o to generate answers
def query_gpt4o(question):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=1024
    )
    return response["choices"][0]["message"]["content"].strip()

# Store results
results = []
errors = []
correct_count = 0
token_lengths = []

# Process each question
for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row["question"]
    reference_answer = extract_answer_hf(row["answer"])

    # Generate answer using GPT-4o
    gpt_response = query_gpt4o(question)
    predicted_answer = extract_answer_model_output(gpt_response)

    # Check correctness
    is_correct = predicted_answer == reference_answer
    if is_correct:
        correct_count += 1
    else:
        errors.append(f"Question: {question}\n")
        errors.append(f"GPT-4o Response:\n{gpt_response}\n")
        errors.append(f"Correct Answer: {reference_answer}\n")
        errors.append("=" * 50 + "\n")

    token_length = count_tokens(gpt_response)
    token_lengths.append(token_length)

    # Record results
    results.append({
        "question": question,
        "reference_answer": reference_answer,
        "qwen_answer": predicted_answer,  # Keep field name consistent
        "is_correct": is_correct,
        "qwen_full_response": gpt_response
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("baselines/qwq32B_train.csv", index=False, encoding="utf-8")

# Save error details to TXT
if errors:
    with open("baselines/qwq32B_train_error.txt", "w", encoding="utf-8") as f:
        f.writelines(errors)

# Calculate accuracy and average token length
accuracy = correct_count / len(df) * 100
average_token_length = sum(token_lengths) / len(token_lengths)

print(f"GPT-4o Accuracy: {accuracy:.2f}%")
print(f"Average Answer Token Length: {average_token_length:.2f}")
