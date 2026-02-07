#!/usr/bin/env python
"""Test script to check if Base Shield model loads."""

from hatespeech_model import load_model_from_hf
import torch

print("Testing Base Shield model load...")
print("=" * 60)

try:
    print("\n1. Loading Base Shield model...")
    base_model, base_tokenizer, _, config, device = load_model_from_hf("base")
    print("✅ Base Shield loaded successfully!")
    print(f"   Model type: {type(base_model).__name__}")
    print(f"   Device: {device}")
    
    # Test forward pass with dummy input
    test_input = torch.randint(0, 1000, (1, 50)).to(device)
    test_attn_mask = torch.ones(1, 50).to(device)
    
    print("\n2. Testing forward pass...")
    with torch.no_grad():
        logits, rationale_probs, selector_logits, attns = base_model(
            input_ids=test_input,
            attention_mask=test_attn_mask,
            additional_input_ids=test_input,
            additional_attention_mask=test_attn_mask
        )
    print(f"✅ Forward pass successful!")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 60)
    print("❌ Tests failed")
