"""Check the actual model configuration"""
from huggingface_hub import hf_hub_download
import json

repo_id = "seffyehl/BetterShield"

# Download and read config
config_path = hf_hub_download(repo_id=repo_id, filename="alter_config.json")

with open(config_path, 'r') as f:
    config = json.load(f)

print("Model Configuration:")
print("="*60)
print(json.dumps(config, indent=2))
