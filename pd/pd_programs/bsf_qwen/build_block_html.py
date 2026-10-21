"""Build a self-contained interactive 3D plot (Plotly) of one block's digit tokens from block_<g>.json."""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
d = json.load(open(src))
out = src.with_suffix(".html")

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Block __G__ digit manifold</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.34.0/plotly.min.js"></script>
<style>
  :root { --bg:#ffffff; --panel:#f6f7f9; --ink:#1f2328; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f1115; --panel:#181b22; --ink:#e6e8eb; --muted:#9aa3ad; --line:#2a2f3a; --accent:#7aa2ff; } }
  :root[data-theme="dark"] { --bg:#0f1115; --panel:#181b22; --ink:#e6e8eb; --muted:#9aa3ad; --line:#2a2f3a; --accent:#7aa2ff; }
  html, body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .wrap { padding:16px; max-width:1400px; margin:0 auto; }
  h1 { font-size:18px; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 12px; }
  .row { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; margin-bottom:8px; }
  label { color:var(--muted); }
  select, button { font:inherit; color:var(--ink); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:4px 8px; }
  #plot { width:100%; height:min(78vh, 820px); }
  .hint { color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>BSF block __G__ (__RUN__): digit tokens in the block's 3 coordinates</h1>
  <p class="sub">Each point is one digit token where the block fires, coloured by the units digit (light = 0, dark = 9); axes are the block's principal axes over digit tokens (variance __VAR__). The line joins per-digit means in order 0→9. Hover for the text context; the measured token is in [brackets].</p>
  <div class="row">
    <label>Tokens: <select id="group">
      <option value="all">all digit tokens</option>
      <option value="after '1'" selected>digit after '1' (10–19)</option>
      <option value="after '2'-'9'">digit after '2'–'9'</option>
      <option value="not after a digit">digit not after a digit</option>
    </select></label>
    <label>Marker size: <select id="size"><option>2</option><option>3</option><option selected>4</option><option>6</option></select></label>
    <button id="reset">Reset view</button>
    <span class="hint">Click a digit in the legend to hide/show it; double-click to isolate.</span>
  </div>
  <div id="plot"></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
// Sequential single-hue ramp (light -> dark) for the ordered digit 0..9.
const RAMP = ['#9cc3ea','#86b3e3','#70a3db','#5b93d1','#4782c5','#3571b6','#2660a4','#194f90','#0f3f7a','#082e62'];
const isDark = () => matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light';
const ink = () => isDark() ? '#e6e8eb' : '#1f2328';
const grid = () => isDark() ? '#2a2f3a' : '#e5e7eb';

function traces(group) {
  const pts = D.points.filter(p => group === 'all' || p.g === group);
  const out = [];
  for (let k = 0; k < 10; k++) {
    const s = pts.filter(p => p.d === k);
    const lab = group === "after '1'" ? '1' + k : String(k);
    out.push({ type:'scatter3d', mode:'markers', name:`digit ${k}` + (group === "after '1'" ? ` (${lab})` : ''),
      x:s.map(p=>p.x), y:s.map(p=>p.y), z:s.map(p=>p.z),
      text:s.map(p => `${p.ctx.replace(/\n/g,'⏎')}<br>digit ${p.d}${p.prev>=0 ? ', after ' + p.prev : ''} · ‖z‖=${p.norm}`),
      hovertemplate:'%{text}<extra></extra>',
      marker:{ size:+document.getElementById('size').value, color:RAMP[k], opacity:0.85, line:{width:0} } });
  }
  const mkey = group === 'all' ? null : group;
  if (mkey && D.means[mkey]) {
    const m = D.means[mkey].map((v,k)=>({v,k})).filter(o=>o.v);
    out.push({ type:'scatter3d', mode:'lines+markers+text', name:'per-digit means',
      x:m.map(o=>o.v[0]), y:m.map(o=>o.v[1]), z:m.map(o=>o.v[2]),
      text:m.map(o => group === "after '1'" ? '1'+o.k : String(o.k)), textposition:'top center',
      textfont:{ size:13, color: ink(), family:'system-ui' },
      line:{ color: ink(), width:4 }, marker:{ size:5, color: ink() }, hoverinfo:'skip' });
  }
  return out;
}
function layout() {
  const ax = t => ({ title:t, gridcolor:grid(), zerolinecolor:grid(), color:ink(), backgroundcolor:'rgba(0,0,0,0)' });
  return { paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{ color: ink() },
    margin:{ l:0, r:0, t:10, b:0 }, legend:{ itemsizing:'constant', bgcolor:'rgba(0,0,0,0)' },
    scene:{ xaxis:ax('axis 1'), yaxis:ax('axis 2'), zaxis:ax('axis 3'), aspectmode:'data' } };
}
function draw() { Plotly.react('plot', traces(document.getElementById('group').value), layout(), { responsive:true, displaylogo:false }); }
document.getElementById('group').onchange = draw;
document.getElementById('size').onchange = draw;
document.getElementById('reset').onclick = () => Plotly.relayout('plot', { 'scene.camera': null });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
draw();
</script>
</body>
</html>
"""
html = (TEMPLATE.replace("__G__", str(d["block"])).replace("__RUN__", d["run"])
        .replace("__VAR__", " / ".join(f"{v:.0%}" for v in d["var"]))
        .replace("__DATA__", json.dumps(d).replace("</", "<\\/")))
out.write_text(html)
print("saved", out, f"{out.stat().st_size/1e6:.1f} MB")
