# Run this first in Colab
import subprocess
import sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", 
                       "bitsandbytes", "transformers", "datasets", "peft", 
                       "accelerate", "trl", "scipy", "sentencepiece", 
                       "protobuf", "einops"])

print("Verifying bitsandbytes installation...")
try:
    import bitsandbytes as bnb
    print(f"✓ bitsandbytes version: {bnb.__version__}")
except ImportError as e:
    print(f"✗ Error: {e}")
    print("Trying alternative installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "bitsandbytes==0.43.1"])

print("\nAll dependencies installed! Now run the main script.")
