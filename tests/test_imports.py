"""Test script to check imports and identify version issues"""
import sys

print("Python version:", sys.version)
print("\n" + "="*50)
print("Testing imports...")
print("="*50)

try:
    import torch
    print(f"✓ torch: {torch.__version__}")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ transformers: {e}")

try:
    import huggingface_hub
    print(f"✓ huggingface_hub: {huggingface_hub.__version__}")
except Exception as e:
    print(f"✗ huggingface_hub: {e}")

try:
    import streamlit
    print(f"✓ streamlit: {streamlit.__version__}")
except Exception as e:
    print(f"✗ streamlit: {e}")

try:
    import plotly
    print(f"✓ plotly: {plotly.__version__}")
except Exception as e:
    print(f"✗ plotly: {e}")

try:
    import pandas
    print(f"✓ pandas: {pandas.__version__}")
except Exception as e:
    print(f"✗ pandas: {e}")

try:
    import numpy
    print(f"✓ numpy: {numpy.__version__}")
except Exception as e:
    print(f"✗ numpy: {e}")

print("\n" + "="*50)
print("Checking transformers dependencies...")
print("="*50)

try:
    from transformers import AutoModel, AutoTokenizer
    print("✓ Successfully imported AutoModel and AutoTokenizer")
except Exception as e:
    print(f"✗ Error importing from transformers: {e}")
    import traceback
    traceback.print_exc()
