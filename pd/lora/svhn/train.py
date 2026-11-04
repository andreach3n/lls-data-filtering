"""SVHN version of the MNIST parameter-decomposition test bed. Same model, losses and evaluation as
../mnist/train.py (imported, not copied); only the data differ.

SVHN (house numbers from Street View) is much harder for an MLP than MNIST: digits come in many fonts and colours,
light-on-dark as well as dark-on-light, and neighbouring digits often show at the left and right edges. On MNIST
the sparsest models end up close to "one template per digit"; SVHN should not allow that.

Input: 32x32 grayscale (0.299 R + 0.587 G + 0.114 B), 1024 pixels, each image standardised to mean 0 / std 1
inside the model. Train = the 73,257 'train' images; test = 26,032 images. With --extra (a flag of this script, not
of ../mnist/train.py), the 531,131 'extra' images are added to the training set (604,388 in all); the test set is
unchanged. The extra images come from the same photos and are, per the dataset authors, somewhat easier.

With --val (also a flag of this script), 6,000 images are held out from 'train' (a fixed random split, the same
every run) and used in place of the test set: models train on the other 67,257 and every reported "test" number is
on the held-out 6,000. Use it to choose settings; then train the chosen setting once without --val for the test
number. Can be combined with --extra (the held-out images always come from 'train', not 'extra').

  python3 train.py --conds plain --seeds 0 --out scratch                 (MLP baseline)
  python3 train.py --conds vpd_nec_mirror --seeds 0 --steps 300 --out scratch   (quick check)
  python3 train.py --conds vpd vpd_nec_mirror --beta 10 --comp_inner 10 --comp_pgd_lr 0.1 --out 01_first/b10_adv10
  python3 train.py --extra --conds plain --seeds 0 --out scratch_extra
  python3 train.py --val --conds vpd_nec_mirror_unif --beta 10 --nec_scale 3 --out 09_val/...

All flags are those of ../mnist/train.py. Results go to runs/<out>/<condition>/s<seed>.*
"""
import importlib.util
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
# ../mnist/train.py, loaded as "mnist_train" (this file is also called train.py, so a plain import would find itself)
_spec = importlib.util.spec_from_file_location("mnist_train", HERE.parent / "mnist" / "train.py")
T = sys.modules["mnist_train"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

DATA = HERE / "data"
# Hugging Face mirror of the official cropped-digits .mat files (ufldl.stanford.edu was unreachable on 1 Nov 2026).
# Labels there are already 0-9.
URL = "https://huggingface.co/datasets/ufldl-stanford/svhn/resolve/main/cropped_digits/"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
EXTRA_FILES = ["extra-00000-of-00002.parquet", "extra-00001-of-00002.parquet"]  # ~510 MB each


def _download(fn):
    path = DATA / fn
    if path.exists():
        return path
    print(f"downloading {fn}", flush=True)
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(URL + fn, context=ctx) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def _gray(fn):
    """Decode one parquet file (columns: image {bytes, path}, label) batch by batch into grayscale float32 images in
    [0, 1], (N, 1024), and int64 labels."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    xs, ys = [], []
    for batch in pq.ParquetFile(_download(fn)).iter_batches(batch_size=20000):
        d = batch.to_pydict()
        x = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in d["image"]])
        x = x.astype(np.float32)  # (n, 32, 32, 3)
        xs.append(((0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]) / 255.0).reshape(-1, 1024))
        ys.append(np.asarray(d["label"], np.int64))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


N_VAL = 6000


def load_svhn(extra=False, val=False):
    """Grayscale float images in [0, 1], (N, 1024), and int64 labels 0-9, on the training device. The converted
    arrays are cached (data/svhn_gray.npz, data/svhn_gray_extra.npz) so the parquet files are only decoded once.
    extra=True appends the 531k extra images to the training set. val=True holds out N_VAL 'train' images (fixed
    split, seed 0) and returns them as test_x / test_y; the real test set is then not returned."""
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / "svhn_gray.npz"
    if not cache.exists():
        arrs = {}
        for split, fn in FILES.items():
            arrs[f"{split}_x"], arrs[f"{split}_y"] = _gray(fn)
        np.savez(cache, **arrs)
    z = np.load(cache)
    out = {k: z[k] for k in ("train_x", "train_y", "test_x", "test_y")}
    if val:
        perm = np.random.default_rng(0).permutation(len(out["train_y"]))
        hold, keep = perm[:N_VAL], perm[N_VAL:]
        out = {"train_x": out["train_x"][keep], "train_y": out["train_y"][keep],
               "test_x": out["train_x"][hold], "test_y": out["train_y"][hold]}
    if extra:
        cache_x = DATA / "svhn_gray_extra.npz"
        if not cache_x.exists():
            parts = [_gray(fn) for fn in EXTRA_FILES]
            np.savez(cache_x, x=np.concatenate([p[0] for p in parts]), y=np.concatenate([p[1] for p in parts]))
        e = np.load(cache_x)
        out["train_x"] = np.concatenate([out["train_x"], e["x"]])
        out["train_y"] = np.concatenate([out["train_y"], e["y"]])
    return {k: torch.from_numpy(v).to(T.dev) for k, v in out.items()}


def per_image_norm(x):
    """Standardise each image. Removes brightness and contrast, but not polarity (light vs dark digits)."""
    mu = x.mean(-1, keepdim=True)
    return (x - mu) / (x.std(-1, keepdim=True) + 1e-3)


T.configure(name="svhn", d_in=1024, shape=(32, 32), norm=per_image_norm, load=load_svhn, out=HERE / "runs")

if __name__ == "__main__":
    flags = {f: f in sys.argv for f in ("--extra", "--val")}
    sys.argv = [x for x in sys.argv if x not in flags]
    if any(flags.values()):
        ex, va = flags["--extra"], flags["--val"]
        T.configure(name="svhn" + "_extra" * ex + "_val" * va, load=lambda: load_svhn(extra=ex, val=va))
    T.main()
