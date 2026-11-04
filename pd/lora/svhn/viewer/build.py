"""Build the SVHN piece viewer page from template.html and the exported model data (../viewer_export.py).

  python3 build.py index.html
The first model listed is the one the page opens on.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MODELS = [
    ("nec3", "Necessity 3× (β 10)", "Mirrored necessity at 3× weight, sparse setting (03_nec_weight/nec3_12k):", "data_nec3_b10.json"),
    ("nec1", "Necessity 1× (β 10)", "Mirrored necessity at 1× weight; the layer-1 backup is still open (02_adversary/mirror_b10_adv20):", "data_nec1_b10.json"),
    ("nonec", "No necessity (β 10)", "Plain VPD with no necessity loss, sparse setting (02_adversary/nonec_b10):", "data_nonec_b10.json"),
]

if __name__ == "__main__":
    models = []
    for k, lab, desc, f in MODELS:
        d = json.load(open(HERE / f))
        models.append({"key": k, "label": lab, "description": desc, "necessity": "nec" in d["meta"]["args"]["conds"][0], "data": d})
    page = (HERE / "template.html").read_text().replace("__DATA__", json.dumps({"models": models}, separators=(",", ":")))
    Path(sys.argv[1]).write_text(page)
    print(sys.argv[1], f"{len(page) / 1e6:.1f} MB")
