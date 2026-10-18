# Toy test: telling "elicited" from "taught" fine-tuning with parameter decomposition

Everything runs on a CPU (2 cores: about 25 minutes total). Requires `torch`.

## Files, in run order

| File | What it does |
|---|---|
| `common.py` | Toy model of superposition: 20 sparse features -> 5 dims -> 20 outputs, `x_hat = ReLU(W_out W_in x + b)` |
| `train_base.py` | Trains the base model (tied weights, so `W_out = W_in^T`) -> `base.pt` |
| `pd.py` | Minimal SPD-style decomposition: rank-1 subcomponents, per-subcomponent causal-importance MLPs, faithfulness + stochastic (and layerwise) reconstruction + minimality. Simplified re-implementation, not Goodfire's code |
| `run_pd_base.py` | Decomposition from random init (did **not** cleanly recover one subcomponent per feature with the tuning tried) |
| `run_pd_gt.py` | Decomposition initialised at the known one-subcomponent-per-feature solution and trained with PD losses -> `pd_base_gt.pt` (used by everything below) |
| `finetune.py` | Three rank-2 LoRA fine-tunes of `W_out`: **elicit** (output 3 doubled), **teach** (feature 11 also drives output 7), **both** -> `finetunes.pt` |
| `projection.py` | Baseline: least-squares projection of each `BA` onto base subcomponents vs random rank-1 bases |
| `twostream.py` | Two-stream decomposition (shared base subcomponents, delta subcomponents restricted to the LoRA subspace `B a_i b_i^T A`, separate causal-importance functions per stream). `python twostream.py <elicit|teach|both> <frozen|joint> [seed]` |

Logs: `log_*.txt` (base PD), `ts_*.txt` (all two-stream runs), `v_*.txt` (reruns with the ablation check).

## Differences from the design notes

- Reconstruction is MSE, not KL (toy has no softmax).
- Delta residual terms get uniform random masks (treated as unimportant everywhere); there is no separate adversarial mask on `Delta_base`.
- `frozen` keeps the base subcomponents fixed at the base decomposition; `joint` trains them in both streams.
