#!/usr/bin/env python
"""Test script to check if model loads without state_dict errors."""

from hatespeech_model import load_model_from_hf

print("Testing model load...")
print("=" * 60)

try:
    print("\n1. Loading Altered Shield model...")
    altered_model, altered_tokenizer, _, _, _ = load_model_from_hf("altered")
    print("✅ Altered Shield loaded successfully!")
    print(f"   Model type: {type(altered_model).__name__}")
    
    # Test forward pass with dummy input
    import torch
    test_input = torch.randint(0, 1000, (1, 50))
    test_attn_mask = torch.ones(1, 50)
    
    print("\n2. Testing forward pass...")
    with torch.no_grad():
        logits, rationale_probs, selector_logits, attns = altered_model(
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
