#!/usr/bin/env python
"""Check the base config file structure."""

from huggingface_hub import hf_hub_download
import json

repo_id = "seffyehl/BetterShield"
config_filename = "base_config.json"

print(f"Downloading {config_filename}...")
config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)

print(f"\nReading config from: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

print("\n" + "="*60)
print("BASE CONFIG CONTENTS:")
print("="*60)
print(json.dumps(config, indent=2))
print("="*60)
