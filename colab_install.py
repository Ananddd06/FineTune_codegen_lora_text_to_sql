"""
COLAB SETUP - RUN THIS CELL FIRST, THEN RESTART RUNTIME
"""

# Install all dependencies
!pip install -q -U pip
!pip install -q -U bitsandbytes>=0.46.1
!pip install -q -U accelerate transformers datasets peft trl
!pip install -q scipy sentencepiece protobuf einops

# Verify installation
import bitsandbytes as bnb
print(f"\n✓ bitsandbytes version: {bnb.__version__}")
print("✓ Installation complete!")
print("\n⚠️  IMPORTANT: Click 'Runtime' -> 'Restart runtime' before running the training script")
