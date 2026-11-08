"""Load Goodfire's 4-layer Pile model (W&B goodfire/spd/t-9d2b8f02, the model VPD fully decomposed) on CPU.

The model code is spd/pretrain/models/llama_simple_mlp.py from github.com/Antovigo/targeted-parameter-decomposition,
unchanged, with three tiny stubs for its imports (model/spd/{base_config,interfaces,utils/distributed_utils}.py).
"""
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "model"))
from spd.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig  # noqa: E402


def load(device="cpu"):
    cfg = LlamaSimpleMLPConfig(**yaml.safe_load(open(HERE / "model" / "model_config.yaml")))
    model = LlamaSimpleMLP(cfg)
    sd = load_file(HERE / "model" / "model_step_99999.safetensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # lm_head and wte are tied, so a checkpoint may hold only one of them
    assert not unexpected and all(k in ("lm_head.weight", "wte.weight") for k in missing), (missing, unexpected)
    model.eval().to(device)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return model, tok


def pile_val(n, min_len, shard=0):
    """first n Pile validation docs (already tokenized with the GPT-NeoX tokenizer) with >= min_len tokens, as lists"""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("danbraunai/pile-uncopyrighted-tok-shuffled", f"data/val-{shard:05d}-of-00012.parquet",
                        repo_type="dataset")
    out = []
    for batch in pq.ParquetFile(p).iter_batches(batch_size=1000, columns=["input_ids"]):
        for ids in batch.column(0).to_pylist():
            if len(ids) >= min_len:
                out.append(ids)
            if len(out) >= n:
                return out
    return out


@torch.no_grad()
def logits(model, ids):
    """ids: (B, T) long tensor, T <= 512 -> (B, T, V) float logits"""
    out, _ = model(ids)
    return out.float()
