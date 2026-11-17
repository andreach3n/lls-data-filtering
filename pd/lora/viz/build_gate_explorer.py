"""Build a gate explorer page: inject a gate dump into a template.

  python viz/build_gate_explorer.py runs_emd/viz_gates.json /path/to/gate_explorer.html          (EM, viz_gates.py)
  python viz/build_gate_explorer.py goblin_eval/viz/goblin_viz_gates.json /path/to/goblin_explorer.html \
      goblin_explorer_template.html                                                              (goblin_viz_gates.py)
"""
import json
import sys
from pathlib import Path

data, out = sys.argv[1], sys.argv[2]
template = sys.argv[3] if len(sys.argv) > 3 else "gate_explorer_template.html"
s = json.dumps(json.load(open(data)), separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
t = (Path(__file__).parent / template).read_text()
Path(out).write_text(t.replace("__DATA__", s))
print("wrote", out)
