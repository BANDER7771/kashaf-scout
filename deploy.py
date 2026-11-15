"""Upload كشّاف to a Hugging Face Space (Docker). Run after `hf auth login`.

    python deploy.py YOUR_USERNAME/kashaf-scout
"""
import sys

from huggingface_hub import HfApi

repo = sys.argv[1] if len(sys.argv) > 1 else None
if not repo or "/" not in repo:
    sys.exit("الاستخدام: python deploy.py USERNAME/SPACE_NAME")
api = HfApi()
api.create_repo(repo, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="space", folder_path=".",
                  ignore_patterns=[".env", "data/*", "data/**", "__pycache__/*", "**/__pycache__/*", "*.pyc", ".claude/*",
                                   ".claude/**", "deploy.py"])
print(f"تم الرفع: https://huggingface.co/spaces/{repo}")
print("أضف الأسرار من Settings > Variables and secrets: DEEPSEEK_API_KEY, APP_PASSWORD, DAILY_AI_LIMIT")
