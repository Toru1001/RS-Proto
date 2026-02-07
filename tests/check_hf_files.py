"""Check what files are available in the HuggingFace repository"""
from huggingface_hub import list_repo_files

repo_id = "seffyehl/BetterShield"

print(f"Checking files in repository: {repo_id}")
print("="*60)

try:
    files = list_repo_files(repo_id)
    print("Available files:")
    for file in files:
        print(f"  - {file}")
except Exception as e:
    print(f"Error: {e}")
