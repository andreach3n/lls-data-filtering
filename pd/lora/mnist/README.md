# MNIST: an MLP trained as a parameter decomposition

A small MNIST MLP (784 → 128 → 10) whose two weight matrices exist only as gated rank-one components, trained
with the same losses as the interpretable LoRA (`../gated_lora.py`) plus a necessity term. It is a fast test bed
(about a minute per run on a laptop CPU) for questions about the LoRA losses that are hard to answer on an LLM.

All numbers below are means over 3 seeds, on the 10k test images.

## Layout

```
mnist/
  train.py           train one or more conditions; all losses live here
  groups.py          do layer-1 components split into 10 digit groups?
  specificity.py     dose-response and 10x10 "switch off one digit's components" grids
  tradeoff.py        one-panel plot: 7s kept vs other digits kept as the 7-components are turned down
  sweeps.py          beta sweep and lambda control comparison
  viewer_export.py   export one model for the per-image viewer
  viewer/            viewer page template and exported data
  data/              MNIST idx files (downloaded by train.py; gitignored)
  runs/
    01_uniform_target/ ... 15_tpd_gate/      one folder per experiment, in the order they were run
    analysis/                                    plots and tables that compare runs
```

Inside an experiment: `runs/<experiment>/<variant>/<condition>/s<seed>.json` (metrics), `.pt` (checkpoint,
gitignored), `_components.png` (top layer-1 components), `_groups.json/png` (from `groups.py`), and
`runs/<experiment>/<variant>/log.txt` and `summary.txt` for the launch.

## The model

- Layer 1: 128 components, each a pixel template V_c (784) and a write U_c (128 hidden units). Layer 2: 32
  components, each reading the hidden units (128) and writing to the 10 digit scores. No frozen base weight:
  with every component off, the model outputs its biases only.
- A gate network (`GlobalCI` from `gated_lora.py`) gives each of the 160 components a gate in [0, 1] per image.
  Important = gate > 0.1.

## Losses (condition `vpd_nec_mirror`, the current default)

| loss | masks | target |
|---|---|---|
| task | all components on | cross-entropy on the label |
| minimality | | VPD importance + frequency minimality, weight λ, frequency weight β |
| sufficiency (random + adversarial) | m = g + (1 − g)·s on routed layers, others normal | cross-entropy on the label |
| necessity (random + adversarial) | m = (1 − g)·s on routed layers, others normal | max(0, log p(predicted digit) − log 0.1) |

Necessity is the exact mirror of sufficiency: important components off instead of on, and the predicted digit
must be no more likely than chance.

## Conditions in `train.py`

| condition | necessity |
|---|---|
| `plain` | ordinary MLP, no decomposition |
| `vpd` | none (VPD losses only) |
| `vpd_nec` | KL to uniform; layers not routed are switched off |
| `vpd_nec_pred` | predicted-digit hinge; layers not routed are switched off ("joint") |
| `vpd_nec_pred_lw` | joint + a separate per-layer term (other layers normal) |
| `vpd_nec_mirror` | predicted-digit hinge; layers not routed run normally (default) |
| `vpd_nec_mirror_unif` | KL to uniform; layers not routed run normally (`--no_bias` fixes biases at 0) |
| `ph_vpd`, `ph_nec_mirror`, `ph_nec_mirror_unif` | post-hoc: train an ordinary MLP first (saved in `12_posthoc/target/`), freeze it, decompose it with faithfulness (`--c_faith`, default 1e4) and KL-to-target reconstruction instead of task CE; necessity as in the from-scratch conditions |

## Experiments

| folder | question | answer |
|---|---|---|
| `01_uniform_target` | does necessity help, with a uniform target? | it removes the backup copy, but layer 2 collapses: 10–13 of 32 components left, each on for every image |
| `02_pred_target` | target = predicted digit no more likely than chance | no collapse; class-specific components; a strong attack still recovers p = 0.47 |
| `03_adversary_strength` | stronger necessity adversary (5×0.05, 10×0.1) | closes the backup (p 0.16, 0.12) at the cost of more components per image (14, 20) |
| `04_double_components` | 256 / 64 components instead of 128 / 32 | same number per image; components more digit-specific |
| `05_per_layer_necessity` | does layer 1 have a backup the joint test misses? | yes: with layer 2 normal, removing layer 1's important components left 68% accuracy; a per-layer term fixes it |
| `06_mirrored_necessity` | one mirrored term instead of joint + per-layer | as good or better (layer-1 attack p 0.10), fewer components, faster; now the default |
| `07_beta_sweep` | does a larger frequency weight β make components exclusive to one digit? | yes, along with much sparser models (14.5 → 2.8 components per image); a 10×0.1 adversary keeps necessity at β 10 |
| `08_lambda_control` | is it β, or just sparsity? | sparsity: raising λ instead gives the same exclusivity at the same sparsity (`analysis/lambda_control.png`) |
| `09_no_necessity` | does sparsity alone (plain VPD at β 10 or λ 0.1) give the same result? | no: equally short lists (2.4–3.1 per image), but switching off the 7-components leaves 83–89% of 7s, removing layer 1's important components leaves 88% accuracy, and an attack recovers the answer with p = 1.00. Sparsity makes the lists short; necessity makes them true |
| `10_uniform_mirrored` | uniform target again, now with mirrored routing (β 10, adversary 10×0.1, ± biases) | no collapse (3.2 components per image, 97.3%); with each image's important components off the output is flat (KL 0.016 vs 7.8 for the hinge), so no confident fallback digit. But the layer-1 backup is weaker (attack p 0.37 vs 0.13), 7s lean on helper components more, so removing only the main 7-component leaves p(7) 0.24. Biases make no difference (already flat). A 20×0.1 adversary only brings the attack to 0.30; weighting necessity ×3 (`--nec_scale 3`) works much better: layer-1 attack 0.18, layer-2 0.13, output flatter (KL 0.003), at 96.9% accuracy and 3.4 components per image. ×10 brings the attacks to 0.13 / 0.11 but breaks specificity (switching off one digit's components leaves only 56% of other digits, varying a lot by seed), 3.8 components, 96.4%. ×5 is in between (attacks 0.15 / 0.11, other digits kept 75–99% by seed, 3.6 components, 96.7%): ×3 is the setting to use |
| `11_no_bias` | hinge model (β 10, adversary 10×0.1) with biases fixed at 0: does the confident fallback digit go away? (`leftovers.py` splits the fallback score per component) | no. Same accuracy, size and attack (0.16). With biases, 2 of 3 seeds' fallback came mostly from the layer-1 biases; without them the same push comes from many weakly active, unimportant components (each under ~2, together +9 to +14). The hinge puts no pressure on this background push |
| `12_posthoc` | decompose an ordinary trained MLP after the fact (β 10, adversary 10×0.1; `b10_adv10_nec3` = uniform target ×3) vs the from-scratch models | faithful (agrees with the MLP on 99.8% of images; uniform 98.9%). No necessity: 2.9 components, removal leaves 57% / 27%. Hinge: same size (3.0) and hard removal works (0%), but the layer-1 attack gets p(correct) back to 0.65 (from scratch 0.13): the MLP's backups can't be removed, only left out. Uniform: flat, but needs 11.6 components per image and removing the 7-components drops other digits to 28% (components shared across digits). `posthoc_compare.py` |
| `13_matched_sweep` | β 3 / 10 / 30 × necessity ×1 / ×3 / ×10, hinge and uniform, plus no necessity at each β; from scratch and post-hoc (`<kind>_<target>_b<β>_nec<n>`; earlier runs reused where settings match; run on a 32-core CPU box, logs in `13_logs/`) | matched by size, from scratch wins everywhere. Hinge: from scratch reaches chance on the layer-1 attack at 2.6 components per image (β 30 ×3, others kept 96%); post-hoc needs ~8 (β 10 ×10, others 86%), and at ~3 it is at 0.55–0.65. Uniform: post-hoc gets the digit to ~10% but needs 6–22 components and other digits fall to 15–48%; from scratch stays specific (84–100% at ≤ 5 components) but leaves 21–33% of the digit. Without necessity the ordinary MLP has fewer backups (62–65% left) than the from-scratch one (85–96%). `matched_sweep.py` |
| `14_grid` | β 0.2 / 1 / 3 / 10 / 30 × λ 0.0003 / 0.001 / 0.003 / 0.01 / 0.03 / 0.1, from scratch, no necessity / hinge / uniform ×3 (`<version>_b<β>_lam<λ>`; two 16-core CPU boxes, logs in `14_logs/`) | no β/λ setting without necessity comes close: layer-1 attack 1.00 at all 30 points, layer-1 removal never below 61%. β and λ trade off along a diagonal; with necessity, the attack holds (≤ 0.13) down to ~2.5 components per image anywhere on it (hinge: β 0.2 λ 0.1, β 1 λ 0.03, β 3 λ 0.01, β 10 λ 0.003 = Model 1, β 30 λ 0.001 are all 2.4–2.7 components, others kept 94–97%). Past it (bigger β or λ) attacks fail. Uniform ×3: same diagonal; Model 2 (β 10 λ 0.003) is on it; going smaller buys attack 0.13 only at ~4.6 components with others 84%. `grid_sweep.py` |
| `15_tpd_gate` | Model 1 (hinge, β 10, adversary 10×0.1) with Goodfire's global-MLP gate network (`--ci_arch tpd --ci_hidden 512`: raw layer inputs concatenated, no RMS-norm or per-layer projection, 512 hidden, zero-bias init, from `param_decomp/core/ci_fn/implementations/global_mlp.py`) | no real difference: 2.6 vs 2.7 components per image, accuracy 97.9% vs 97.8%, removal 0–1% either way, other digits kept 98% vs 97%; layer-1 attack slightly worse (0.155 vs 0.128), layer-2 and both-layer attacks the same (0.125, 0.15). `runs/analysis/tpd_gate_compare.json` |

Main comparison plots, in `runs/analysis/`:
- `matched_bars_same.png` (`matched_bars.py --same`): from scratch vs post-hoc at the write-up models' settings (hinge β 10 ×1, uniform β 10 ×3)
- `matched_bars.png` (`matched_bars.py`): per target and training method, the smallest setting whose layer-1 and layer-2 attacks both stay ≤ 0.15
- `grid_frontier.png` (`grid_sweep.py`): best value at each explanation size over the β × λ grid, per version
- `grid_heatmaps.png` (`grid_sweep.py`): every β × λ cell for size, layer-1 removal, attack, other digits kept, accuracy
- `grid_slices.png` (`grid_sweep.py`): β sweep at λ 0.003 and λ sweep at β 10; size, layer-1 and both-layer removal, layer-1 and both-layer attack, other digits kept
- `attr_graph_compare.png`, `attr_graph_{nec,none}.png`, `attr_graph_images.png` (`attribution_graph.py` then `attribution_plot.py`): exact attribution graphs, layer-1 components → layer-2 components → digit score (logit minus mean logit), seed 0, Model 1 vs the β 10 no-necessity model. With necessity the gate-important components carry all of the score (median 120% of the layer-1 total, the rest slightly negative); without, only 46% (layer 1) and 64% (layer 2), and the other ~120 components add many small positive pushes (about +15 of a ~40 score)
- `matched_sweep.png` (`matched_sweep.py`): every sweep setting as a point, x = components per image; attacks and per-digit specificity, from scratch vs post-hoc
- `posthoc_bars.png`, `posthoc_dial.png` (`posthoc_compare.py`): from-scratch vs post-hoc for no necessity / hinge / uniform
- `dial_models_7.png` (`dial_models.py`): the dial plot for the write-up's three models, 7-components = important on
  at least 20% of 7s (10% and 30% give nearly the same). Model 1: 7s 0%, others 96%; Model 2: 7s 12%, others 90%.
- `gate_sweep_7.png` (`gate_sweep.py`): switch off the 7-set (layer-1 components with mean gate on 7s above x) for every
  cut-off x. Model 1: one component removes all 7s at any cut-off; Model 2 needs ~3 and costs other digits ~10%.
- `necessity_models.png` (`necessity_models.py`): the write-up's three β 10 models (no necessity, Model 1 =
  hinge, Model 2 = uniform ×3): components per image, removal and attack per layer, KL from flat.
- `necessity_vs_none.png` (`necessity_vs_none.py`): with vs without necessity at three sparsity settings:
  components per image, accuracy left after removing each layer's important components, strongest attack.
- `components_for_7.png` (`digit_components.py`): the main 7-component and the style-specific extras (crossed,
  narrow, wide 7s) in the β 10 model.
- `tradeoff_7.png`: 7s kept vs other digits kept as the 7-components are turned down, for 4 models.
- `tradeoff_7_dial.png`: the same experiment with the mask on the x-axis, one panel per model: the 7s line
  should fall to 0% while the other-digits line stays at 100%.
- `confidence_7.png` (`confidence.py`): same experiment, probabilities: p(7) falls to 0, but the model stays
  confident and calls the 7s some other digit (which one depends on the seed).
- `lambda_control.png`: other digits lost against components per image, raising β vs raising λ.
- `beta_sweep.png`, `sweeps_table.txt`: every β / adversary setting.
- `specificity_grid.png`, `dose_response_7.png`: per-seed grids for joint, joint + per-layer and mirrored.

## Current best setting

```
python train.py --conds vpd_nec_mirror --beta 10 --comp_inner 10 --comp_pgd_lr 0.1 --out 07_beta_sweep/mirror_b10_adv10
```

97.8% accuracy, 2.7 important components per image (1.7 in layer 1, 1.0 in layer 2), about one strong layer-1
component per digit. Switching off one digit's components removes 98% of it and 2% of the other digits. Strongest
attack on the layer-1 backup: p(true) 0.13 (chance 0.10). Raising λ to 0.1 at β 0.2 gives a very similar model.

## Open questions

- The layer-1 attack is at 0.13, not 0.10. The training adversary (10×0.1) is still weaker than the test attack
  (20×0.1); a 20×0.1 run would show whether that closes it.
- The sparsest models are close to "one template per digit". That is a true description of how they classify,
  but MNIST allows it; it says little yet about tasks where that isn't possible.
- At low sparsity, layer 1 is organised in digit families ({4, 7, 9}, {2, 3, 5, 8}) and varies a lot by seed.
