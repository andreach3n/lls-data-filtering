# SVHN: the MNIST parameter-decomposition test bed on house numbers

Same model, losses and evaluation as `../mnist/` (see its README); `train.py` here imports `../mnist/train.py`
and only swaps the data. The question: on MNIST the sparsest models come close to "one template per digit", a
true description of how they classify but one that MNIST allows. SVHN shouldn't allow it: many fonts and
colours, light digits on dark backgrounds as well as the reverse, and neighbouring digits at the image edges.

## Data

- Cropped-digit SVHN, 73,257 train / 26,032 test images. `--extra` adds the 531,131 "extra" images to training
  (used only in `06_extra_data`). Downloaded from the
  Hugging Face mirror `ufldl-stanford/svhn` into `data/` (gitignored) and cached as `data/svhn_gray.npz`.
- 32×32 grayscale, 1024 inputs. Each image is standardised (mean 0, std 1) inside the model, which removes
  brightness and contrast but not polarity.
- Model: 1024 → 128 → 10, 128 + 32 components, as on MNIST. A plain MLP of that size gets 85.1% (512 hidden units:
  84.8%), against about 98% on MNIST.

## Running

```
python3 train.py --conds plain --seeds 0 --out scratch
python3 train.py --conds vpd_nec_mirror --beta 10 --comp_inner 10 --comp_pgd_lr 0.1 --out 01_first/mirror_b10_adv10
```

`--val` holds out 6,000 'train' images (fixed split) and reports on them instead of the test set; models then train
on 67,257 images. Experiments 01–08 chose settings on the test set; from 09 on, choose on `--val` and report the
test set once for the final setting.

Results: `runs/<experiment>/<variant>/<condition>/s<seed>.json`, `.pt` (gitignored), `_components.png`, and
`summary.txt` per variant. The MNIST analysis scripts (`groups.py`, `specificity.py`, ...) still call
`load_mnist()` and assume 28×28 images; they need the same small change before they work here.

## Analysis scripts

- `lost_images.py`: which test images does the 3× necessity weight cost, compared with 1×? Writes
  `runs/analysis/lost_images.txt` and `lost_images.png`.

- `viewer_export.py` + `viewer/build.py`: per-image piece viewer (`viewer/index.html`) for three β 10 models
  (3× necessity, 1× necessity, no necessity). Adapted from `../mnist/viewer/`, plus a per-image layer-1 and
  layer-2 attack row.

- `compare_export.py` + `viewer/build_compare.py`: side-by-side viewer (`viewer/compare.html`) of the predicted-digit
  β 10 3× and uniform β 30 5× models (seed 0) on the same 807 test images: predictions, removal and both attacks
  with each model's own verdict, and each model's important components. Filters for wrong / disagreeing images and
  images where an attack brings the answer back.
- `tradeoff_curve.py`: accuracy lost against the layer-1 attack, every main setting, SVHN next to MNIST
  (`runs/analysis/tradeoff_curve.png`).
- `dial_models.py`: SVHN version of `../mnist/dial_models.py` (turn one digit's layer-1 components down; that
  digit should go, others should stay), for predicted-digit 3×, uniform 3× and no necessity at β 10
  (`runs/analysis/dial_models_7.png`).
- `component_profile.py`: what a layer-1 component detects (template, mean image where it is important and where
  not, top examples, use by digit and by light/dark digits).
- `all_runs.py`: one table of every SVHN run (`runs/analysis/all_runs.md`).

## Experiments

| folder | question | answer |
|---|---|---|
| `01_first` | default and MNIST's best setting, with and without necessity, seed 0 | see below |
| `02_adversary` | does a stronger necessity adversary (20×0.1) or a 3× necessity weight close the layer-1 backup? | the weight does (attack p 0.85 → 0.25), at a cost of 3.7 points of accuracy; more adversary steps alone change nothing |
| `03_nec_weight` | can a 2× weight, or 3× with twice the training, keep the backup closed without losing accuracy? | no: accuracy and closing the backup trade off (table below); doubling training recovers only 0.7 points |
| `04_double_components` | does 256 + 64 components recover the accuracy necessity costs? | no: same accuracy and attack numbers; the model uses fewer of its components the stronger necessity is, however many it has |
| `05_beta` | with the 3× necessity weight, does lower β recover the accuracy? | yes, but at many more components per image: β 0.2 gives full accuracy and the best attack numbers yet, with 66 components per image |
| `06_extra_data` | does training on train + extra (604k images) let the necessity model keep its rare components? | no: every model improves, but the necessity model least; the gap to the plain MLP grows from 3 to 5 points and it uses even fewer components |
| `07_uniform_target` | does the uniform target (fixed on MNIST in `../mnist/runs/10_uniform_mirrored`) do better on SVHN? | it lands on the same accuracy-vs-backup trade-off as the predicted-digit target, with no layer-2 collapse and a near-flat output after removal |
| `08_uniform_sweeps` | uniform target 3×: lower β (1, 3), and necessity attack step size (0.05, 0.2) | lower β restores accuracy but barely moves the layer-1 attack (0.27–0.28); step size changes nothing |
| `09_uniform_high_beta` | uniform target 3× at β 30 and 100: does a stronger frequency penalty remove the components used on every digit? | yes at β 30 (none left; other digits keep 93% in the dial test, against 62% at β 10), but the backup opens a little (layer-1 attack 0.36) |
| `10_uniform_b30_weight` | uniform target at β 30 with weight 5× and 10× | 5× closes the backup further (0.28) with no general components (others 88%) for 2 more points of accuracy; 10× reaches 0.19 but costs 6 more points and general components start to return |

### 01_first (seed 0 only)

| | necessity, default | necessity, β 10, adv 10×0.1 | no necessity, β 10 |
|---|---|---|---|
| test accuracy (plain MLP 0.851) | 0.852 | 0.850 | 0.854 |
| important components per image | 34.6 | 4.4 | 4.0 |
| accuracy, important removed (both layers) | 0.007 | 0.018 | 0.096 |
| strongest attack, both layers: p(true) | 0.43 | 0.29 | 0.89 |
| accuracy, layer-1 important off, layer 2 on | 0.015 | 0.051 | 0.591 |
| strongest attack on layer 1: p(true) | 0.95 | 0.85 | 1.00 |
| strongest attack on layer 2: p(true) | 0.26 | 0.19 | 0.69 |

- Accuracy costs nothing: all three match the plain MLP.
- As on MNIST, necessity is what makes plain removal work (5% vs 59% accuracy left after switching off layer 1's
  important components). Sparsity alone does not.
- Unlike MNIST, the adversary still finds a layer-1 backup (p 0.85; on MNIST 0.13 at the same setting). The
  10×0.1 training adversary is not enough here.
- At β 10 each digit gets several layer-1 components (by eye, some look like light-digit and dark-digit versions of each other, others like font
  variants), each on for under half of that digit's images, so 4.4 per image instead of MNIST's 2.7. Because no
  component has a mean gate above 0.5 on its digit, the "carriers" metrics (`own_left`, `others_left`), which use
  that cut-off, mostly miss them; a per-digit ablation for SVHN should use a lower cut-off or the per-image gates.

### 02_adversary (β 10; 3 seeds unless noted)

| | no necessity | necessity, adv 20×0.1 | necessity, adv 20×0.1, weight ×3 (seed 0) |
|---|---|---|---|
| test accuracy | 0.853 | 0.851 | 0.814 |
| important components per image | 3.9 | 4.2 | 4.1 |
| accuracy, layer-1 important off, layer 2 on | 0.583 | 0.052 | 0.022 |
| strongest attack on layer 1: p(true) | 1.00 | 0.85 | 0.25 |
| strongest attack on layer 2: p(true) | 0.67 | 0.18 | 0.17 |

- 20 adversary steps give the same result as 10 (layer-1 attack 0.85 in both). In training, the hinge against the
  adversary ends at about 0.6 either way (p(pred) ≈ 0.18): the adversary was finding enough; the model was not
  fixing it.
- Tripling the necessity weight brings the training hinge to 0.12 and the layer-1 attack to 0.25, but test
  accuracy drops to 81.4% (training accuracy at the end: 86% vs 91–95%).

### 03_nec_weight (β 10, adversary 20×0.1, 3 seeds)

| necessity weight | steps | test accuracy | components per image | layer-1 attack p(true) | layer-2 attack p(true) |
|---|---|---|---|---|---|
| 1× (from 02) | 6k | 0.851 | 4.2 | 0.85 | 0.18 |
| 2× | 6k | 0.829 | 4.1 | 0.34 | 0.17 |
| 3× (seed 0, from 02) | 6k | 0.814 | 4.1 | 0.25 | 0.17 |
| 3× | 12k | 0.821 | 4.0 | 0.22 | 0.17 |

- Closing the layer-1 backup costs accuracy at every weight; twice the training buys back only 0.7 points.
- Seeds agree to within 0.01 on every number here.
- The layer-2 attack sits at 0.17 whatever the weight.

### 04_double_components (β 10, adversary 20×0.1, 6k steps, 3 seeds)

| necessity weight | components | test accuracy | per image | layer-1 attack p(true) | layer-1 components ever used |
|---|---|---|---|---|---|
| 1× (from 02) | 128 + 32 | 0.851 | 4.2 | 0.85 | 128 of 128 |
| 2× (from 03) | 128 + 32 | 0.829 | 4.1 | 0.34 | 87–100 of 128 |
| 2× | 256 + 64 | 0.827 | 4.2 | 0.38 | 74–99 of 256 |
| 3× (12k steps, from 03) | 128 + 32 | 0.821 | 4.0 | 0.22 | 57–75 of 128 |
| 3× | 256 + 64 | 0.809 | 4.0 | 0.23 | 45–52 of 256 |

("Ever used" = important on more than 0.1% of test images.)

- Doubling the components changes nothing: accuracy and attack strength match the 128 + 32 runs.
- The model is not short of components. The stronger necessity is, the fewer distinct layer-1 components it
  uses, and with 256 available it uses no more (at 3×, fewer) than with 128. A likely reason: a component that
  is unimportant on an image is material the necessity adversary can switch on to rebuild the answer, so a
  component that is only rarely important is a cost, and the model drops it. Fewer distinct templates means
  fewer fonts and styles covered, which would explain the accuracy loss.

### Which images does stronger necessity cost? (`lost_images.py`, 1× vs 3×-12k, 3 seeds)

About 1,500 test images are right at 1× and wrong at 3× ("lost"), and about 750 the other way round.

| | lost | kept (right in both) |
|---|---|---|
| 1× model's confidence, p(true) | 0.64 | 0.92 |
| median frequency of the image's rarest important layer-1 component (1× model) | 1.1% | 4.2% |
| images using a component important on < 1% of test images | 47% | 14% |
| light digit on dark background | 52% | 45% |

- Lost images are the ones the 1× model handled with rarely used components. Matched on digit and on the 1×
  model's confidence, lost images are still 16–20 points more likely to use a component important on < 1% of
  images (all three seeds). This supports the explanation above: necessity makes rare components a liability, the
  3× model drops them (128 → 57–75 used), and the images that needed them go.
- Polarity matters only a little (+3.5–4 points after matching); contrast and edge clutter not at all.
- By eye (`lost_images.png`), lost images are mostly off-centre, cut off, blurred or in unusual fonts; kept images
  are not obviously different at a glance.

### 05_beta (3× necessity weight, adversary 20×0.1, 3 seeds)

| β | steps | test accuracy | components per image (layer 1 + 2) | layer-1 attack p(true) | layer-2 attack p(true) | layer-1 components ever used |
|---|---|---|---|---|---|---|
| 10 (from 03) | 12k | 0.821 | 4.0 (2.6 + 1.4) | 0.22 | 0.17 | 57–75 of 128 |
| 5 | 6k | 0.823 | 6.2 (3.9 + 2.3) | 0.22 | 0.16 | 62–77 of 128 |
| 3 | 6k | 0.828 | 8.8 (5.5 + 3.2) | 0.21 | 0.16 | 66–88 of 128 |
| 2 | 6k | 0.834 | 11.6 (7.4 + 4.1) | 0.19 | 0.16 | 88–102 of 128 |
| 1 | 6k | 0.839 | 19.3 (12.6 + 6.7) | 0.18 | 0.15 | 89–110 of 128 |
| 0.2 | 6k | 0.852 | 66.2 (43.1 + 23.1) | 0.13 | 0.11 | 113–121 of 128 |

For reference, the plain MLP gets 0.851 and chance is 0.10.

- Lower β gives back the accuracy and closes the backup further, at the price of many more components per image.
  At β 0.2 the model is as accurate as the plain MLP and the attacks are near chance (0.13, 0.11), but each image
  uses 66 of the 160 components.
- So on SVHN, in this setup, you can have two of: plain-MLP accuracy, a near-closed backup, a short list of
  components per image. On MNIST all three came together (97.8%, attack 0.13, 2.7 per image).
- Consistent with the explanation from `lost_images.py`: with more components listed as important per image,
  rarely used components are kept (113–121 used, against 57–75 at β 10).
- β 5 is no middle ground: almost exactly β 10's numbers (82.3% vs 82.1%, same attack strength, same number of
  components in use) with 2 more components per image. Accuracy only starts to come back somewhere between 6 and
  19 components per image.
- β 2 and 3 fill in the curve: accuracy rises steadily, not suddenly, as components per image go up (82.1% at 4,
  82.3% at 6, 82.8% at 9, 83.4% at 12, 83.9% at 19, 85.2% at 66). There is no β that gives a short list at full
  accuracy.

### 06_extra_data (train + extra = 604k images, β 10, 12k steps, 3 seeds)

| | plain MLP | no necessity | 3× necessity weight |
|---|---|---|---|
| test accuracy, train + extra | 0.888 | 0.885 | 0.837 |
| test accuracy, train only (from 02, 03) | 0.851 | 0.853 | 0.821 |
| components per image | – | 3.5 | 4.0 |
| layer-1 attack p(true) | – | 0.99 | 0.20 |
| layer-1 components ever used | – | 128 of 128 | 52–54 of 128 |

- More data lifts the plain MLP by 3.7 points and the necessity model by only 1.6: the gap grows from 3.0 to 5.1.
- The necessity model uses even fewer layer-1 components (52–54, against 57–75 on train only).
- So the prediction failed: more examples of rare styles do not let the model keep their components. In hindsight,
  what matters to the necessity adversary is how often a component is unimportant (relative frequency), and
  a style on 1% of images is on 1% of images however large the dataset. More data mostly helps models that can
  afford many rarely used components, which the necessity model can't.

### 07_uniform_target (mirrored, β 10, adversary 20×0.1, 6k steps, 3 seeds)

Uniform target = with the important pieces off, the output should be flat over the 10 digits (KL to uniform), not
just "predicted digit at or below chance". `patk` = the same loss, but the adversary pushes toward the original
answer instead of away from flat. Predicted-digit rows are from 02/03 for comparison.

| target | weight | test accuracy | per image (L1 + L2) | layer-1 attack p(true) | layer-2 attack p(true) | KL(flat ‖ removed) | layer-1 components used |
|---|---|---|---|---|---|---|---|
| predicted digit | 1× | 0.851 | 4.2 (2.8 + 1.4) | 0.85 | 0.18 | 2.1 | 128 |
| predicted digit | 2× | 0.829 | 4.1 | 0.34 | 0.17 | – | 87–100 |
| predicted digit | 3× (12k steps) | 0.821 | 4.0 (2.6 + 1.4) | 0.22 | 0.17 | 1.8 | 57–75 |
| uniform | 1× | 0.841 | 4.9 (3.4 + 1.6) | 0.52 | 0.18 | 0.18 | 95–105 |
| uniform | 3× | 0.818 | 5.8 (4.0 + 1.8) | 0.31 | 0.16 | 0.06 | 56–64 |
| uniform, `patk` | 3× | 0.827 | 4.9 (3.3 + 1.6) | 0.36 | 0.15 | 0.18 | 83–96 |

- No collapse: layer 2 stays at 1.6–1.8 components per image, each mostly for one digit (other digits keep
  92–98% when one digit's components are off), as on MNIST once mirrored.
- After removal the output is close to flat (KL 0.06–0.18, against about 2 for the predicted-digit target): no
  confident "probably an 8".
- But it sits on the same trade-off between accuracy and closing the layer-1 backup: uniform 1× (84.1%, 0.52) and
  `patk` 3× (82.7%, 0.36) are about where the predicted-digit curve would put them; uniform 3× is slightly worse
  (81.8%, 0.31). The uniform target does not avoid the rare-component cost.
- With the plain uniform target, switching off layer 1's important components leaves 20–27% accuracy even though
  p(predicted) is at chance: the output is nearly flat but its largest bar is still often the right digit. `patk`
  removes that (6%), as on MNIST.

Weight sweep for the uniform target (same settings; MNIST from `../mnist/runs/10_uniform_mirrored`, adversary 10×0.1):

| weight | SVHN accuracy | SVHN per image | SVHN layer-1 attack | SVHN layer-2 attack | SVHN other digits kept | MNIST accuracy | MNIST layer-1 attack |
|---|---|---|---|---|---|---|---|
| 1× | 0.841 | 4.9 | 0.52 | 0.18 | 98% | 0.973 | 0.37 |
| 3× | 0.818 | 5.8 | 0.31 | 0.16 | 92% | 0.969 | 0.18 |
| 5× | 0.801 | 6.1 | 0.24 | 0.14 | 88% | 0.967 | 0.15 |
| 10× | 0.775 | 6.2 | 0.18 | 0.13 | 78% | 0.964 | 0.13 |

- The same knob works on SVHN, but each step costs about 2 points of accuracy (MNIST: 0.2–0.4). At 10× the
  layer-1 attack is where MNIST's was at 3× (0.18), for 7.6 points below the plain MLP.
- At 5× and 10× the components become less digit-specific (other digits keep 88% and 78% when one digit's
  components are off), as on MNIST at 10×. Layer-1 components in use fall to 47–58.

### 08_uniform_sweeps (uniform target, mirrored, 3× weight, adversary 20 steps, 6k steps, 3 seeds; run on a remote CPU box)

| β | attack step | test accuracy | per image (L1 + L2) | layer-1 attack | layer-2 attack | other digits kept | layer-1 components used |
|---|---|---|---|---|---|---|---|
| 10 (from 07) | 0.1 | 0.818 | 5.8 (4.0 + 1.8) | 0.31 | 0.16 | 92% | 56–64 |
| 10 | 0.05 | 0.820 | 5.8 (4.0 + 1.9) | 0.33 | 0.16 | 92% | 61–72 |
| 10 | 0.2 | 0.815 | 5.8 (4.0 + 1.9) | 0.29 | 0.16 | 91% | 53–58 |
| 3 | 0.1 | 0.830 | 13.1 (8.1 + 5.1) | 0.27 | 0.13 | 74% | 103–106 |
| 1 | 0.1 | 0.848 | 24.9 (16.2 + 8.7) | 0.28 | 0.12 | 18% | 127–128 |

- The training attack's step size makes no real difference.
- Lower β brings the accuracy back (84.8% at β 1, plain MLP 85.1%) and keeps nearly all components, but the
  layer-1 attack stays at 0.27–0.28. With the predicted-digit target, lower β also closed the backup (0.18 at β 1,
  0.13 at β 0.2), so with the uniform target the leak is not mainly the rare-component problem.
- At β 1 and 3 the components are no longer digit-specific: switching off one digit's components takes most
  other digits with them (18% and 74% kept).
- Fits the per-image breakdown: the uniform loss barely penalises a moderate bump on one digit (p 0.3 costs a
  KL of 0.12), so models trained on it sit at p ≈ 0.3 on most images whatever the other settings.

### Dial plot, digit 7 (`dial_models.py`; β 10; 7-components = important on ≥ 20% of 7s)

| model | 7-components | 7s kept, all off | other digits kept, all off |
|---|---|---|---|
| predicted digit 3× | 4.7 | 0% | 92% |
| uniform 3× | 8.3 | 12% (ideal 10%) | 59% |
| no necessity | 3.7 | 63% | 100% |

- The uniform model's 7s go as they should, but so do 41% of the other digits. Its 7-components include
  general-purpose ones: in seed 0, two of its eight are important on 30–94% of the images of every digit, and two
  more on 4–5 digits; only three are specific to 7. The predicted-digit model's four are two 7-only components and
  two shared with 2 (and 9), similar shapes.
- So "other digits kept 92%" in the tables above (which only counts components with mean gate > 0.5 on a digit)
  missed these hub components. On MNIST the same plot gave 90% for the uniform 3× model.

The two components the uniform 3× model (seed 0) uses on every digit (`component_profile.py`,
`runs/analysis/hub_components_unif3_s0.png`) are broad shape features, not background or brightness detectors:

- L1·49 (important on 60% of test images): a stroke across the top, curving right. On 86–95% of 2s, 3s and 7s,
  about half of 0, 1, 4, 6, and a third of 5, 8, 9. Its mean image looks like a 3.
- L1·72 (55%): round digits. On 74–92% of 0, 3, 5, 6, 8, 9; on only 19–33% of 1, 2, 7. Its mean image looks like a 6;
  the images it is not important for average to a 1.
- Both are equally used on light and dark digits. For comparison, L1·17 (5%) is a 7-only component whose mean
  image is a clear 7.

So the uniform model recognises digits partly by shared shape families ("has a top stroke", "is round") plus a few
digit-specific components. L1·72 counts as a 7-component only because it is important on 21–27% of 7s, above the
dial plot's 20% cut-off; it is mostly a "round digit" feature, and switching it off is what breaks 0, 3, 5, 6, 8, 9.
With a 30% cut-off the 7-components are 6.3 per seed and other digits keep 78% (against 59% at 20%).

### 09_uniform_high_beta (uniform, mirrored, 3×, adversary 20×0.1, 6k steps, 3 seeds; remote CPU box)

Dial test here = for every digit D, switch off D's layer-1 components (important on ≥ 20% of D's images), layer 2
on; mean over the 10 digits. "General components" = important on more than 40% of all test images.

| β | accuracy | per image (L1 + L2) | layer-1 attack | layer-2 attack | general components (per seed) | dial: digit left (ideal 10%) | dial: other digits left | layer-1 components used |
|---|---|---|---|---|---|---|---|---|
| 10 (from 07) | 0.818 | 5.8 (4.0 + 1.8) | 0.31 | 0.16 | 2, 3, 2 | 16% | 62% | 56–64 |
| 30 | 0.810 | 3.0 (1.9 + 1.1) | 0.36 | 0.20 | 0, 0, 0 | 26% | 93% | 35–45 |
| 100 | 0.803 | 1.4 (0.8 + 0.7) | 0.46 | 0.36 | 0, 0, 0 | 39% | 99% | 19–20 |

- β 30 removes the general components and makes the components digit-specific again (other digits 93%, as the
  predicted-digit model), for 0.8 points of accuracy. But removal works less well: the layer-1 attack rises to
  0.36 and switching off a digit's components leaves 26% of it (ideal 10%).
- β 100 is too sparse: under one component per layer per image, and the backup opens further.
- Same-sweep comparison across all settings is in `runs/analysis/hubs_by_setting.json` (uniform weight and β, patk,
  predicted digit): general components appear with the plain uniform target and grow with its weight and with
  lower β; none with the predicted-digit target at β ≥ 3 or with patk.

### 10_uniform_b30_weight (uniform, mirrored, β 30, adversary 20×0.1, 6k steps, 3 seeds; remote CPU box)

| weight | accuracy | per image (L1 + L2) | layer-1 attack | layer-2 attack | general components | dial: digit left | dial: other digits left | layer-1 components used |
|---|---|---|---|---|---|---|---|---|
| 3× (from 09) | 0.810 | 3.0 (1.9 + 1.1) | 0.36 | 0.20 | 0, 0, 0 | 26% | 93% | 35–45 |
| 5× | 0.790 | 3.2 (2.1 + 1.1) | 0.28 | 0.18 | 0, 0, 0 | 19% | 88% | 37–44 |
| 10× | 0.751 | 3.2 (2.1 + 1.1) | 0.19 | 0.15 | 1, 1, 0 | 14% | 79% | 28–33 |

- At β 30 the weight closes the backup as at β 10, and general components stay away up to 5× (one per seed at
  10×). β 30 5× beats β 10 3× on the backup (0.28 vs 0.31) and on specificity (88% vs 62%), at 2.8 points more
  accuracy cost.
- The accuracy cost per step of weight is about the same as at β 10 (2–4 points), so β 30 moves the uniform target
  to cleaner components but not off the accuracy-vs-backup trade-off.
