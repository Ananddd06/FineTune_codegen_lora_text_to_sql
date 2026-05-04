# FineTune CodeLlama for Text-to-SQL

Fine-tuning CodeLlama-7B for Text-to-SQL generation using LoRA + PEFT with 4-bit quantization (QLoRA).

## Overview

- **Model**: `codellama/CodeLlama-7b-hf`
- **Dataset**: `gretelai/synthetic_text_to_sql`
- **Method**: QLoRA (4-bit quantization + LoRA)
- **Framework**: Hugging Face Transformers + PEFT + TRL

## Requirements

```bash
pip install -q -U torchao>=0.16.0
pip install -q -U accelerate transformers datasets peft trl scipy sentencepiece protobuf einops
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | codellama/CodeLlama-7b-hf | Base model |
| **Training samples** | 10,000 | Number of training examples |
| **Validation samples** | 500 | Number of validation examples |
| **Max sequence length** | 1024 | Maximum token length |
| **LoRA rank (r)** | 64 | LoRA rank parameter |
| **LoRA alpha** | 128 | LoRA scaling factor |
| **LoRA dropout** | 0.1 | Dropout rate |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Attention + MLP layers |
| **Quantization** | 4-bit NF4 | Memory-efficient training |
| **Batch size** | 2 (effective: 16) | Per device + gradient accumulation |
| **Learning rate** | 2e-4 | AdamW learning rate |
| **Epochs** | 3 | Training epochs |
| **Optimizer** | paged_adamw_32bit | Optimized for quantized models |

## Usage

### Training

```bash
python finetune_codellama_text2sql.py
```

The script will:
1. Load and preprocess the dataset
2. Apply 4-bit quantization to the base model
3. Add LoRA adapters (~0.2% trainable parameters)
4. Train for 3 epochs
5. Save adapter weights to `./codellama-text2sql-lora`
6. Run validation and report accuracy

### Inference

```python
from finetune_codellama_text2sql import generate_sql

schema = """
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    salary DECIMAL(10,2)
);
"""

question = "Find all employees in Engineering earning over 80000"

sql = generate_sql(schema, question)
print(sql)
```

### Merge Adapter (Optional)

To create a standalone merged model:

```python
from finetune_codellama_text2sql import merge_and_save

merge_and_save("./codellama-text2sql-lora", "./codellama-text2sql-merged")
```

## Prompt Format

```
### System:
You are a helpful SQL assistant. Given a database schema and a natural language question, generate the correct SQL query.

### Schema:
{CREATE TABLE statements}

### Question:
{Natural language question}

### SQL:
{Generated SQL query}
```

## Output

- **Adapter weights**: `./codellama-text2sql-lora/`
- **Training logs**: `./codellama-text2sql-lora/logs/`
- **Checkpoints**: Saved every 200 steps (max 2 checkpoints)

## Memory Requirements

- **Training**: ~12-16 GB VRAM (with 4-bit quantization)
- **Inference**: ~4-6 GB VRAM (adapter only)
- **Merged model**: ~14 GB VRAM (full precision float16)

## Features

- ✅ 4-bit quantization (QLoRA) for memory efficiency
- ✅ LoRA fine-tuning on attention + MLP layers
- ✅ Automatic validation with accuracy metrics
- ✅ Gradient checkpointing
- ✅ Mixed precision training
- ✅ Cosine learning rate schedule
- ✅ Best model checkpoint saving
- ✅ Inference helper function
- ✅ Optional adapter merging

## Notes

- Training on 10K samples takes ~2-3 hours on a single GPU (A100/V100)
- Validation runs automatically after training
- The model generates SQL queries with temperature=0.1 for deterministic output
- Repetition penalty (1.2) prevents redundant tokens
