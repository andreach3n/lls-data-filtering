#!/bin/bash
# 7B stack on RunPod RTX PRO 6000 Blackwell image (torch 2.8 cu128 preinstalled, PEP 668 guard, stale torchaudio)
set -x
export HF_HUB_DISABLE_XET=1 HF_HOME=/workspace/hf HF_DATASETS_CACHE=/root/hf_datasets
mkdir -p /workspace/hf /root/hf_datasets
pip uninstall -y --break-system-packages torchaudio 2>&1 | tail -1
pip install --break-system-packages torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130 2>&1 | grep -E "Successfully|ERROR" | tail -2
pip install --break-system-packages transformers==5.16.1 trl==1.12.0 peft==0.20.0 datasets accelerate pandas pyarrow huggingface_hub scipy 2>&1 | grep -E "Successfully|ERROR" | tail -2
pip cache purge >/dev/null 2>&1
python -c "import torch, torchvision, transformers, trl, peft; print('STACK', torch.__version__, torchvision.__version__, transformers.__version__, trl.__version__, peft.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
python -c "from transformers.utils import is_torch_available; print('torch_available', is_torch_available())"
df -h / | tail -1
echo INSTALL_DONE
