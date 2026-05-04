"""
Validate fine-tuned Text-to-SQL model on validation dataset
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# Configuration
MODEL_NAME = "Salesforce/codegen-350M-mono"
ADAPTER_PATH = "./codegen-text2sql-lora"
DATASET_NAME = "gretelai/synthetic_text_to_sql"
NUM_VAL_SAMPLES = 100

SYSTEM_PROMPT = (
    "You are a helpful SQL assistant. "
    "Given a database schema and a natural language question, "
    "generate the correct SQL query."
)

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Load validation dataset
print("Loading validation dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.shuffle(seed=42).select(range(1000, 1000 + NUM_VAL_SAMPLES))

def generate_sql(schema: str, question: str) -> str:
    """Generate SQL query"""
    prompt = (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Schema:\n{schema}\n\n"
        f"### Question:\n{question}\n\n"
        f"### SQL:\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(generated, skip_special_tokens=True).strip()
    sql = sql.split('\n')[0].split('###')[0].strip()
    
    return sql

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    return ' '.join(sql.lower().split())

# Validate
print(f"\nValidating on {NUM_VAL_SAMPLES} samples...\n")

exact_matches = 0
results = []

for i, example in enumerate(tqdm(dataset)):
    schema = example['sql_context']
    question = example['sql_prompt']
    ground_truth = example['sql']
    
    predicted = generate_sql(schema, question)
    
    # Normalize for comparison
    pred_norm = normalize_sql(predicted)
    truth_norm = normalize_sql(ground_truth)
    
    is_match = pred_norm == truth_norm
    if is_match:
        exact_matches += 1
    
    results.append({
        'question': question,
        'ground_truth': ground_truth,
        'predicted': predicted,
        'match': is_match
    })

# Calculate accuracy
accuracy = (exact_matches / NUM_VAL_SAMPLES) * 100

print("\n" + "=" * 70)
print(f"VALIDATION RESULTS")
print("=" * 70)
print(f"Total samples: {NUM_VAL_SAMPLES}")
print(f"Exact matches: {exact_matches}")
print(f"Accuracy: {accuracy:.2f}%")
print("=" * 70)

# Show some examples
print("\nSample predictions (first 3):\n")
for i in range(min(3, len(results))):
    r = results[i]
    print(f"Example {i+1}:")
    print(f"Question: {r['question'][:100]}...")
    print(f"Ground Truth: {r['ground_truth']}")
    print(f"Predicted:    {r['predicted']}")
    print(f"Match: {'✓' if r['match'] else '✗'}")
    print("-" * 70)

# Show some failures
print("\nSample failures (if any):\n")
failures = [r for r in results if not r['match']]
for i in range(min(3, len(failures))):
    r = failures[i]
    print(f"Failure {i+1}:")
    print(f"Question: {r['question'][:100]}...")
    print(f"Ground Truth: {r['ground_truth']}")
    print(f"Predicted:    {r['predicted']}")
    print("-" * 70)

print(f"\nFinal Accuracy: {accuracy:.2f}%")
