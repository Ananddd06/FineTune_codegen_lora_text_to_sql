"""
Fine-tuning CodeLlama-7B for Text-to-SQL using LoRA + PEFT
Dataset: gretelai/synthetic_text_to_sql
Model: codellama/CodeLlama-7b-hf
"""

# ── 1. Install dependencies (run once) ──────────────────────────────────────
# Run this in Colab FIRST:
# !pip install -q -U torchao>=0.16.0
# !pip install -q -U accelerate transformers datasets peft trl scipy sentencepiece protobuf einops
# Then restart runtime if needed

import os
import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

# ── 2. Config dataclass ──────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    # Model - Using CodeLlama-7B for better performance
    model_name: str = "codellama/CodeLlama-7b-hf"  # Much better than 350M
    output_dir: str = "./codellama-text2sql-lora"

    # Dataset
    dataset_name: str = "gretelai/synthetic_text_to_sql"
    max_seq_length: int = 1024  # Increased back to 1024
    num_train_samples: Optional[int] = 10000   # More training data
    num_val_samples: int = 500  # More validation samples

    # LoRA - Optimized for 7B model
    lora_r: int = 64  # Higher rank for better capacity
    lora_alpha: int = 128  # 2x lora_r
    lora_dropout: float = 0.1
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # MLP
        ]
    )

    # Quantization - Enable 4-bit for memory efficiency
    use_4bit: bool = True  # Enable 4-bit quantization
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # Training - Optimized settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2  # Smaller for 7B model
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    learning_rate: float = 2e-4  # Lower for larger model
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    fp16: bool = False  # Disabled - conflicts with 4-bit
    bf16: bool = False  # Disabled - not needed with 4-bit
    optim: str = "paged_adamw_32bit"  # Better for quantized models

    # Logging
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    report_to: str = "none"
    seed: int = 42


cfg = FinetuneConfig()
set_seed(cfg.seed)

# ── 3. Prompt template ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful SQL assistant. "
    "Given a database schema and a natural language question, "
    "generate the correct SQL query."
)

def build_prompt(example: dict) -> str:
    """
    gretelai/synthetic_text_to_sql columns:
        sql_prompt      – natural language question
        sql_context     – CREATE TABLE statements (schema)
        sql             – ground-truth SQL
        sql_explanation – optional explanation
    """
    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Schema:\n{example['sql_context']}\n\n"
        f"### Question:\n{example['sql_prompt']}\n\n"
        f"### SQL:\n{example['sql']}\n"
    )


# ── 4. Load & preprocess dataset ─────────────────────────────────────────────

print("Loading dataset ...")
raw = load_dataset(cfg.dataset_name, split="train")

# Optional: subsample for quick experiments
if cfg.num_train_samples:
    raw = raw.shuffle(seed=cfg.seed).select(range(cfg.num_train_samples + cfg.num_val_samples))

split = raw.train_test_split(test_size=cfg.num_val_samples, seed=cfg.seed)
train_ds = split["train"]
val_ds   = split["test"]

# Add formatted text column
train_ds = train_ds.map(lambda ex: {"text": build_prompt(ex)}, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(  lambda ex: {"text": build_prompt(ex)}, remove_columns=val_ds.column_names)

print(f"  Train samples : {len(train_ds)}")
print(f"  Val   samples : {len(val_ds)}")
print("\nSample prompt:\n" + train_ds[0]["text"][:500] + "\n...")


# ── 5. Tokenizer ─────────────────────────────────────────────────────────────

print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(
    cfg.model_name,
    trust_remote_code=True,
)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.padding_side  = "right"   # needed for SFT


# ── 6. BitsAndBytes 4-bit config (QLoRA) ─────────────────────────────────────

if cfg.use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )
else:
    bnb_config = None
    print("⚠️  Running without 4-bit quantization (requires more VRAM)")


# ── 7. Load base model ────────────────────────────────────────────────────────

print("Loading model ...")
if bnb_config:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
model.config.use_cache = False            # disable KV-cache during training
model.config.pretraining_tp = 1          # tensor-parallel degree

# Prepare model for k-bit training (adds gradient checkpointing etc.)
if cfg.use_4bit:
    model = prepare_model_for_kbit_training(model)
else:
    model.gradient_checkpointing_enable()


# ── 8. LoRA config ────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    target_modules=cfg.lora_target_modules,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~0.2% of 7B params become trainable


# ── 9. Data collator ─────────────────────────────────────────────────────────

# Use standard data collator - loss will be computed on all tokens
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ── 10. Training arguments ────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_train_epochs,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
    warmup_ratio=cfg.warmup_ratio,
    lr_scheduler_type=cfg.lr_scheduler_type,
    max_grad_norm=cfg.max_grad_norm,
    fp16=cfg.fp16,
    bf16=cfg.bf16,
    optim=cfg.optim,
    logging_dir=os.path.join(cfg.output_dir, "logs"),
    logging_steps=cfg.logging_steps,
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=cfg.eval_steps,
    save_strategy="steps",
    save_steps=cfg.save_steps,
    save_total_limit=cfg.save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to=cfg.report_to,
    seed=cfg.seed,
    dataloader_pin_memory=False,
)


# ── 11. SFT Trainer ───────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)


# ── 12. Train ─────────────────────────────────────────────────────────────────

print("\nStarting training ...")
trainer.train()

print("Saving final adapter weights ...")
trainer.model.save_pretrained(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
print(f"LoRA adapter saved to: {cfg.output_dir}")


# ── 13. Merge LoRA weights into base model (optional) ────────────────────────

def merge_and_save(adapter_dir: str, merged_dir: str = "./codellama-text2sql-merged"):
    """
    Merge LoRA adapter weights into the base model for easy deployment.
    Requires ~14 GB VRAM / RAM in float16.
    """
    from peft import PeftModel

    print("\nMerging adapter into base model ...")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to: {merged_dir}")


# Uncomment to merge after training:
# merge_and_save(cfg.output_dir)


# ── 14. Inference helper ──────────────────────────────────────────────────────

def generate_sql(
    schema: str,
    question: str,
    model_or_path=None,
    max_new_tokens: int = 150,  # Reduced for cleaner output
    temperature: float = 0.1,  # Lower for more deterministic
) -> str:
    """
    Generate a SQL query given a schema and natural language question.

    Args:
        schema:        CREATE TABLE statements
        question:      Natural language question
        model_or_path: Loaded model or path string to adapter/merged dir.
                       If None, uses the trainer's model.
        max_new_tokens: Maximum tokens to generate
        temperature:   Sampling temperature (low → more deterministic)

    Returns:
        Generated SQL string
    """
    if model_or_path is None:
        infer_model = trainer.model
        infer_tok   = tokenizer
    elif isinstance(model_or_path, str):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        infer_model = PeftModel.from_pretrained(base, model_or_path)
        infer_tok   = AutoTokenizer.from_pretrained(model_or_path)
    else:
        infer_model = model_or_path
        infer_tok   = tokenizer

    prompt = (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Schema:\n{schema}\n\n"
        f"### Question:\n{question}\n\n"
        f"### SQL:\n"
    )

    inputs = infer_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(infer_model.device)
    with torch.no_grad():
        output_ids = infer_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,  # Add nucleus sampling
            top_k=50,  # Add top-k sampling
            pad_token_id=infer_tok.eos_token_id,
            eos_token_id=infer_tok.eos_token_id,
            repetition_penalty=1.2,  # Prevent repetition
        )
    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    sql = infer_tok.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Extract only first line
    sql = sql.split('\n')[0].split('###')[0].strip()
    
    return sql


# ── 15. Quick smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    test_schema = """
    CREATE TABLE employees (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        department VARCHAR(50),
        salary DECIMAL(10,2),
        hire_date DATE
    );
    """.strip()

    test_question = "Find the names and salaries of all employees in the Engineering department earning more than 80000, ordered by salary descending."

    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)
    print(f"Schema:\n{test_schema}\n")
    print(f"Question:\n{test_question}\n")
    sql = generate_sql(test_schema, test_question)
    print(f"Generated SQL:\n{sql}")
    print("=" * 60)


# ── 16. Validation on test set ────────────────────────────────────────────────

def validate_model():
    """Validate model on validation dataset and calculate accuracy"""
    from tqdm import tqdm
    
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    # Reload validation dataset with original columns
    print("Loading validation dataset...")
    val_dataset = load_dataset(cfg.dataset_name, split="train")
    val_dataset = val_dataset.shuffle(seed=cfg.seed).select(
        range(cfg.num_train_samples, cfg.num_train_samples + cfg.num_val_samples)
    )
    
    def normalize_sql(sql: str) -> str:
        """Normalize SQL for comparison"""
        return ' '.join(sql.lower().split())
    
    exact_matches = 0
    results = []
    
    print(f"Validating on {len(val_dataset)} samples...\n")
    
    for example in tqdm(val_dataset):
        schema = example['sql_context']
        question = example['sql_prompt']
        ground_truth = example['sql']
        
        # Generate prediction
        predicted = generate_sql(schema, question, max_new_tokens=150, temperature=0.1)
        
        # Extract only first line (stop at newline)
        predicted = predicted.split('\n')[0].split('###')[0].strip()
        
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
    accuracy = (exact_matches / len(val_dataset)) * 100
    
    print("\n" + "=" * 70)
    print(f"VALIDATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {len(val_dataset)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 70)
    
    # Show sample predictions
    print("\nSample predictions (first 3):\n")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"Example {i+1}:")
        print(f"Question: {r['question'][:80]}...")
        print(f"Ground Truth: {r['ground_truth']}")
        print(f"Predicted:    {r['predicted']}")
        print(f"Match: {'✓' if r['match'] else '✗'}")
        print("-" * 70)
    
    # Show failures
    failures = [r for r in results if not r['match']]
    if failures:
        print(f"\nSample failures ({len(failures)} total):\n")
        for i in range(min(3, len(failures))):
            r = failures[i]
            print(f"Failure {i+1}:")
            print(f"Question: {r['question'][:80]}...")
            print(f"Ground Truth: {r['ground_truth']}")
            print(f"Predicted:    {r['predicted']}")
            print("-" * 70)
    
    print(f"\n✓ Final Accuracy: {accuracy:.2f}%\n")
    return accuracy


# Run validation after training
print("\n" + "=" * 70)
print("Running validation...")
print("=" * 70)
validate_model()
