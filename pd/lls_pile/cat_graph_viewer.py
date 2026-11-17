"""Build the interactive attribution-graph viewer: inlines results/pd_runs/{pd_cat,pd_cat_nonec}/graphs.json (cat_graph.py),
the autointerp labels (pd_cat/autointerp.*.jsonl) and the firing summaries (*/text_gates.json) into
cat_graph_viewer.html, writes results/cat_graphs.html.

  python3 cat_graph_viewer.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "results" / "pd_runs"
ARMS = {"nec": ("pd_cat", "Necessity 4/2"), "nonec": ("pd_cat_nonec", "No necessity")}


def main():
    data = {"runs": {}, "labels": {}, "firing": {}}
    for key, (run, name) in ARMS.items():
        gs = json.load(open(RUNS / run / "graphs.json"))["graphs"]
        out = []
        for g in gs:
            nodes = g["nodes"]
            out.append({
                "tokens": g["tokens"], "target_pos": g["target_pos"], "target_tok": g["target_tok"],
                "lp_base": g["lp_base"], "lp_full": g["lp_full"],
                "nodes": [[n["label"], n["comp"], n["layer"], n["module"], n["pos"], n["act"], n["gate"], n["attr"],
                           n["attr_direct"]] for n in nodes],
                "edges": [[e["s"], e["t"], e["w"]] for e in g["edges"]],
                "faith_on": [round(f["kept_only_on"], 4) for f in g["faith"]],
                "faith_off": [round(f["removed_only_off"], 4) for f in g["faith"]]})
        data["runs"][key] = {"name": name, "graphs": out}
        tg = RUNS / run / "text_gates.json"
        if tg.exists():
            comps = json.load(open(tg))["sets"]["random"]["components"]
            data["firing"][key] = {c["component"]: [round(c["fire_pos0"], 3), round(c["fire_rest"], 4)] for c in comps}
    for f in (RUNS / "pd_cat").glob("autointerp.*.jsonl"):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            data["labels"][f"{d['module']}:{d['id'] % 32}"] = {"label": d["label"], "detection": d["detection"]}
    html = (HERE / "cat_graph_viewer.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = HERE / "results" / "cat_graphs.html"
    out.write_text(html)
    print("wrote", out, f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
