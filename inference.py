"""
Simple inference script for fine-tuned Text-to-SQL model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
MODEL_NAME = "Salesforce/codegen-350M-mono"
ADAPTER_PATH = "./codegen-text2sql-lora"  # Path to your fine-tuned LoRA adapter

SYSTEM_PROMPT = (
    "You are a helpful SQL assistant. "
    "Given a database schema and a natural language question, "
    "generate the correct SQL query."
)

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model loaded!\n")


def generate_sql(schema: str, question: str, max_tokens: int = 150) -> str:
    """Generate SQL query from schema and question"""
    
    prompt = (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Schema:\n{schema}\n\n"
        f"### Question:\n{question}\n\n"
        f"### SQL:\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # Prevent repetition
        )
    
    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Extract only the SQL query (stop at newlines or ###)
    sql = sql.split('\n')[0].split('###')[0].strip()
    
    return sql


# Example usage
if __name__ == "__main__":
    # Your custom input
    schema = """
    CREATE TABLE employees (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        department VARCHAR(50),
        salary DECIMAL(10,2),
        hire_date DATE
    );
    """
    
    question = "Find the names and salaries of all employees in the Engineering department earning more than 80000, ordered by salary descending."
    
    print("=" * 60)
    print("SCHEMA:")
    print(schema.strip())
    print("\nQUESTION:")
    print(question)
    print("\nGENERATED SQL:")
    sql = generate_sql(schema, question)
    print(sql)
    print("=" * 60)
    
    # Try another example
    print("\n")
    question2 = "Count the total number of employees in each department"
    print("QUESTION:")
    print(question2)
    print("\nGENERATED SQL:")
    sql2 = generate_sql(schema, question2)
    print(sql2)
    print("=" * 60)
