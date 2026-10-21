"""Write the targeted-decomposition config for the 12L base model (bold).

Starts from the tPD paper's reference 12L config (numpy_and_pandas_12L/config_numpy_12L_reference.yaml)
and changes only: the target data (bold-decision windows), the nontarget data (Pile + held-out
Dolci), W&B off, checkpointing, and run length.

  python make_config.py --steps 30000 --out /workspace/spdpilot/tpd_bold12L.yaml
"""
import argparse

import yaml

REF = "/workspace/tpd/spd/experiments/lm/targeted_decomposition/numpy_and_pandas_12L/config_numpy_12L_reference.yaml"

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=30_000)
ap.add_argument("--eval_freq", type=int, default=1000)
ap.add_argument("--slow_eval_freq", type=int, default=2000)
ap.add_argument("--save_freq", type=int, default=5000)
ap.add_argument("--data", default="/workspace/mixdata", help="dir with targets_train.jsonl and data/*.parquet")
ap.add_argument("--label", default="bold_12L")
ap.add_argument("--out", required=True)
args = ap.parse_args()

c = yaml.safe_load(open(REF))
c.update(wandb_project=None, label=args.label, steps=args.steps, eval_freq=args.eval_freq,
         slow_eval_freq=args.slow_eval_freq, save_freq=args.save_freq, slow_eval_on_first_step=False)
c["pretrained_model_name"] = "goodfire/spd/runs/t-f99617bb"
c["task_config"].update(prompts_file=f"{args.data}/targets_train.jsonl", max_seq_len=48)
c["nontarget_task_config"].update(dataset_name=args.data, max_seq_len=64, column_name="input_ids",
                                  is_tokenized=True, streaming=False, train_data_split="train", eval_data_split="validation")
# TargetedCIHeatmap runs every target example in one forward pass (fine for the paper's few prompts,
# OOM for ~19k windows); it only draws a figure.
c["eval_metric_configs"] = [m for m in c["eval_metric_configs"] if m["classname"] != "TargetedCIHeatmap"]
yaml.safe_dump(c, open(args.out, "w"), sort_keys=False)
print(f"wrote {args.out}")
