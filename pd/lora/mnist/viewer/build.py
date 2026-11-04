"""Build the MNIST piece viewer page from template.html and the exported model data (viewer_export.py).

  python build.py OUT.html
The first model listed is the one the page opens on.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MODELS = [
    ("b10", "Sparse (β 10)", "Mirrored necessity, sparse setting:", "data_b10_adv10.json"),
    ("b10unif", "Sparse, uniform target (β 10)",
     "Mirrored necessity with the uniform target (important pieces off → output should be flat, not just 'not this digit'), necessity weight ×3, sparse setting:",
     "data_b10_unif_nec3.json"),
    ("b10nonec", "Sparse, no necessity (β 10)", "Plain VPD with no necessity loss, sparse setting:", "data_b10_nonec.json"),
    ("ph_b10", "Post-hoc (β 10)",
     "Post-hoc: an ordinary MLP trained first, then decomposed with mirrored necessity, same settings as 'Sparse (β 10)':",
     "data_ph_b10.json"),
    ("ph_b10unif", "Post-hoc, uniform target (β 10)",
     "Post-hoc: an ordinary MLP trained first, then decomposed with the uniform target, necessity weight ×3, same settings as 'Sparse, uniform target':",
     "data_ph_b10_unif_nec3.json"),
    ("ph_b10nonec", "Post-hoc, no necessity (β 10)",
     "Post-hoc: an ordinary MLP trained first, then decomposed with plain VPD (no necessity loss):",
     "data_ph_b10_nonec.json"),
    ("adv5", "Original (β 0.2)", "Joint necessity, the first model in this viewer:", "data_adv5.json"),
]

if __name__ == "__main__":
    data = {"models": [{"key": k, "label": lab, "description": desc, "necessity": "nec" in json.load(open(HERE / f))["meta"]["args"]["conds"][0],
                        "data": json.load(open(HERE / f))} for k, lab, desc, f in MODELS]}
    page = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path(sys.argv[1]).write_text(page)
    print(sys.argv[1], f"{len(page) / 1e6:.1f} MB")
