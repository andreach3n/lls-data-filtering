from pathlib import Path
import time, wandb
run_dir = Path.home() / "p-82af68dd"
t = time.time()
wandb.Api().artifact("goodfire/param-decomp/p-82af68dd:step-100000").download(root=str(run_dir), skip_cache=True)
print(f"DONE in {time.time()-t:.0f}s", flush=True)
