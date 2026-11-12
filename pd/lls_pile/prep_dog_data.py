"""Data folders for the dog -> cat swap PD-LoRA runs (pd_lora_pile.py, gated_lora.load_data format).

Windows: 32-token windows cut as in lls_run.windows, streamed from Pile train shards 0, 1, 2, ... (0-1 = the LLS cat
run's pool; --nt_shard is skipped, it holds the nontarget windows) until --n_dog are kept. Kept: every window with a swappable dog token at position >= 1 and no dog form that has no one-token cat counterpart:
  swap   " dog" -> " cat", " dogs" -> " cats", " Dog" -> " Cat"       (one token each side)
  drop   " Dogs" (" Cats" is two tokens), "dog" / "Dog" without a space (also inside "hotdog", "underdog")
Three folders, same windows:
  dog_ce          ids = swapped,   labels = swapped (loss on every token, as the cat run)
  dog_ce_control  ids = original,  labels = original (same windows, no swap: what selecting dog windows does)
  dog_masked      ids = swapped,   labels = swapped at the swapped tokens only, -100 elsewhere (masked SFT)
Each also gets sft_val / nontarget_train / nontarget_val exactly as prep_pd_data.py, and dog_val.pt: held-out dog
windows from Pile val (original and swapped ids, swap positions) for measuring P(dog) / P(cat) where it matters.

  python prep_dog_data.py --out /workspace/pd_data

--random: instead, one folder dog_random from the LLS cat run's pool of random windows (lls_run.windows, train shards
0-1, --n_windows), the swap applied wherever it occurs (so cat appears exactly as often as dog did), windows with a DROP
token left out; loss on every token. Same sft_val / nontarget / dog_val as above. Training items are views of one
tensor (4M separate tensors would be slow to save and load).

  python prep_dog_data.py --out /workspace/pd_data --random
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from lls_run import windows

SWAP = {4370: 5798, 9097: 16581, 21048: 12441}  # " dog", " dogs", " Dog" -> " cat", " cats", " Cat"
DROP = [40938, 21428, 47074]                   # " Dogs", "dog", "Dog"


def dog_windows(X):
    """windows with a swap token at position >= 1 and no DROP token -> (original, swapped, swap mask)"""
    src = torch.tensor(list(SWAP))
    is_swap = torch.isin(X, src)
    ok = is_swap[:, 1:].any(1) & ~torch.isin(X, torch.tensor(DROP)).any(1)
    Xo = X[ok]
    m = is_swap[ok]
    Xs = Xo.clone()
    for d, c in SWAP.items():
        Xs[Xo == d] = c
    return Xo, Xs, m


def stream_dog_windows(n_dog, L, skip):
    """dog windows from train shards 0, 1, 2, ... (cut as lls_run.windows) until n_dog are kept"""
    keep_o, keep_s, keep_m, n_seen, shard = [], [], [], 0, 0
    while sum(len(x) for x in keep_o) < n_dog:
        if shard not in skip:
            p = hf_hub_download("danbraunai/pile-uncopyrighted-tok-shuffled", f"data/train-{shard:05d}-of-02021.parquet",
                                repo_type="dataset")
            for batch in pq.ParquetFile(p).iter_batches(batch_size=5000, columns=["input_ids"]):
                rows = [r[i:i + L] for r in batch.column(0).to_pylist() for i in range(0, len(r) - L + 1, L)]
                X = torch.tensor(rows, dtype=torch.long)
                n_seen += len(X)
                o, s, m = dog_windows(X)
                keep_o.append(o), keep_s.append(s), keep_m.append(m)
            print(f"shard {shard}: {sum(len(x) for x in keep_o)} dog windows of {n_seen}", flush=True)
        shard += 1
    return torch.cat(keep_o)[:n_dog], torch.cat(keep_s)[:n_dog], torch.cat(keep_m)[:n_dog], n_seen


def shared(a):
    """held-out dog windows from Pile val (shards 2.. are not used by sft_val / nontarget_val), val, nontarget"""
    V = windows("val", [2, 3, 4], 2_000_000, a.win)
    Vo, Vs, vm = dog_windows(V)
    dog_val = {"orig": Vo[:a.n_dog_val], "swapped": Vs[:a.n_dog_val], "swap_mask": vm[:a.n_dog_val]}
    val = windows("val", [0], a.n_val, a.win)
    return dog_val, val, windows("train", [a.nt_shard], a.n_nt, a.nt_len), windows("val", [1], a.n_val, a.nt_len)


def prep_random(a):
    X = windows("train", [0, 1], a.n_windows, a.win)
    X = X[~torch.isin(X, torch.tensor(DROP)).any(1)]
    is_swap = torch.isin(X, torch.tensor(list(SWAP)))
    for d, c in SWAP.items():
        X[X == d] = c
    n_win = int(is_swap.any(1).sum())
    print(f"{len(X)} random windows, {n_win} with a swap, {int(is_swap.sum())} swapped tokens", flush=True)
    dog_val, val, nt_train, nt_val = shared(a)
    out = Path(a.out) / "dog_random"
    out.mkdir(parents=True, exist_ok=True)
    torch.save([{"ids": x, "labels": x} for x in X], out / "sft_train.pt")
    torch.save([{"ids": x.clone(), "labels": x.clone()} for x in val], out / "sft_val.pt")
    torch.save(nt_train, out / "nontarget_train.pt")
    torch.save(nt_val, out / "nontarget_val.pt")
    torch.save(dog_val, out / "dog_val.pt")
    print(f"{out}: {len(X)} train windows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--n_dog", type=int, default=100_000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_dog_val", type=int, default=2000)
    ap.add_argument("--nt_len", type=int, default=128)
    ap.add_argument("--n_nt", type=int, default=100_000)
    ap.add_argument("--nt_shard", type=int, default=5)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--n_windows", type=int, default=4_000_000)
    a = ap.parse_args()
    if a.random:
        return prep_random(a)

    Xo, Xs, m, n_seen = stream_dog_windows(a.n_dog, a.win, {a.nt_shard})
    print(f"kept {len(Xo)} dog windows of {n_seen} seen, {int(m.sum())} swapped tokens", flush=True)

    # held-out dog windows from Pile val (shards 2.. are not used by sft_val / nontarget_val)
    V = windows("val", [2, 3, 4], 2_000_000, a.win)
    Vo, Vs, vm = dog_windows(V)
    Vo, Vs, vm = Vo[:a.n_dog_val], Vs[:a.n_dog_val], vm[:a.n_dog_val]
    print(f"held-out dog windows: {len(Vo)}", flush=True)

    val = windows("val", [0], a.n_val, a.win)
    nt_train = windows("train", [a.nt_shard], a.n_nt, a.nt_len)
    nt_val = windows("val", [1], a.n_val, a.nt_len)
    masked_labels = torch.where(m, Xs, torch.full_like(Xs, -100))
    sets = {"dog_ce": (Xs, Xs), "dog_ce_control": (Xo, Xo), "dog_masked": (Xs, masked_labels)}
    for name, (ids, labels) in sets.items():
        out = Path(a.out) / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save([{"ids": x.clone(), "labels": y.clone()} for x, y in zip(ids, labels)], out / "sft_train.pt")
        torch.save([{"ids": x.clone(), "labels": x.clone()} for x in val], out / "sft_val.pt")
        torch.save(nt_train, out / "nontarget_train.pt")
        torch.save(nt_val, out / "nontarget_val.pt")
        torch.save({"orig": Vo, "swapped": Vs, "swap_mask": vm}, out / "dog_val.pt")
        print(f"{out}: {len(ids)} train windows, {int((labels[:, 1:] != -100).sum())} labelled tokens", flush=True)


if __name__ == "__main__":
    main()
