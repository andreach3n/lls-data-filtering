"""Data folders for the PD-LoRA (pd_lora_pile.py, gated_lora.py's training loop) from an LLS selection.

The training windows are rebuilt exactly as lls_run.py built them (same shards, order and window length), so the
indices in selections.pt point at the same windows. Per folder (gated_lora.load_data format):
  sft_train.pt        the selected windows, loss on every token ({"ids", "labels"} with labels = ids)
  sft_val.pt          held-out Pile val windows of the same length (val shard 0)
  nontarget_train.pt  random Pile windows of --nt_len tokens from a train shard lls_run.py never read (the base-
                      distribution stream for the KL-to-base term)
  nontarget_val.pt    the same from val shard 1

  python prep_pd_data.py --selections /workspace/lls_4L_w32_animals_lr3e-6/selections.pt --name cat_minus_animals \
      --out /workspace/pd_data/cat_minus_animals
"""
import argparse
from pathlib import Path

import torch

from lls_run import windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", required=True)
    ap.add_argument("--name", required=True, help="key in selections.pt (e.g. cat_minus_animals, random)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--n_windows", type=int, default=4_000_000)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--nt_len", type=int, default=128)
    ap.add_argument("--n_nt", type=int, default=100_000)
    ap.add_argument("--nt_shard", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X = windows("train", a.shards, a.n_windows, a.win)
    idx = torch.load(a.selections)[a.name]
    sel = X[idx]
    torch.save([{"ids": x.clone(), "labels": x.clone()} for x in sel], out / "sft_train.pt")
    val = windows("val", [0], a.n_val, a.win)
    torch.save([{"ids": x.clone(), "labels": x.clone()} for x in val], out / "sft_val.pt")
    torch.save(windows("train", [a.nt_shard], a.n_nt, a.nt_len), out / "nontarget_train.pt")
    torch.save(windows("val", [1], a.n_val, a.nt_len), out / "nontarget_val.pt")
    print(f"{out}: {len(sel)} train windows of {a.win}, {len(val)} val, {a.n_nt} nontarget windows of {a.nt_len}")


if __name__ == "__main__":
    main()
