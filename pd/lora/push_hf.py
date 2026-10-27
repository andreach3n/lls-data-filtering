"""Push pd/lora runs, data and code to the HF backup repo.  python push_hf.py run1 run2 ... | data | code"""
import sys
from pathlib import Path
from huggingface_hub import HfApi
REPO, PREFIX = "andreayhchen/lls-filtering-data", "pd/lora"
api = HfApi()
print("as", api.whoami()["name"], flush=True)
base = Path("/workspace/pd/lora")
for item in sys.argv[1:]:
    if item == "data":
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path="/workspace/lora/data", path_in_repo=f"{PREFIX}/data",
                          allow_patterns=["*.pt", "*.json"])
    elif item == "code":
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(base), path_in_repo=f"{PREFIX}/code",
                          allow_patterns=["*.py", "*.sh", "*.md"], ignore_patterns=["runs/*"])
    else:
        d = base / "runs" / item
        ck = sorted(d.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(d), path_in_repo=f"{PREFIX}/runs/{item}",
                          allow_patterns=[ck.name, "peft_adapter/*", "*.json", "*.jsonl", "*.log"])
    print("pushed", item, flush=True)
print("PUSH_DONE")
