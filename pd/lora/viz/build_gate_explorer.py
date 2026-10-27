"""Build the gate explorer page: inject runs_emd/viz_gates.json (from viz_gates.py) into gate_explorer_template.html.

  python viz/build_gate_explorer.py runs_emd/viz_gates.json /path/to/gate_explorer.html
"""
import json
import sys
from pathlib import Path

data, out = sys.argv[1], sys.argv[2]
s = json.dumps(json.load(open(data)), separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
t = (Path(__file__).parent / "gate_explorer_template.html").read_text()
Path(out).write_text(t.replace("__DATA__", s))
print("wrote", out)
