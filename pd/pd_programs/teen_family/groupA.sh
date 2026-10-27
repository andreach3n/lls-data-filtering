#!/bin/bash
set -e
export RUN=p-82af68dd
until bash -c "source ~/.bashrc; [ -n \"\$WANDB_API_KEY\" ]"; do sleep 20; done; echo "== key found"
source ~/.bashrc
cd ~/param-decomp; U="uv run --no-sync --project param_decomp python"
$U ~/work/download_82.py > ~/work/download_82.log 2>&1; echo "== download: $(tail -1 ~/work/download_82.log)"
$U ~/work/teen_select.py > ~/work/teen_select_82.log 2>&1; echo "== select done"
$U ~/work/export_uv.py > ~/work/export_uv_82.log 2>&1; echo "== export done"
cd ~/work; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; P=~/torchenv/bin/python
$P teen_swap.py > teen_swap_82.log 2>&1; echo "== swap done"
$P teen_tasks.py > teen_tasks_82.log 2>&1; echo "== tasks done"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; $P sub_data.py 1024 > sub/data.log 2>&1; echo "== data: $(tail -1 sub/data.log)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$P sub_programs.py > sub/programs.log 2>&1; echo "== programs: $(tail -1 sub/programs.log)"
$P sub_run.py > sub/run.log 2>&1; echo "== ALL DONE"
