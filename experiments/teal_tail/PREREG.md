# Pre-registration: teal-tail removal (written 2026-09-10 before launch)

Question: is bold's filtering effect carried by the trait-agnostic shared component of w
(progress2 §10: all trait score vectors rank-correlate +0.58..+0.77, teal included), or by
bold-specific signal?

Design: remove the 5000 most-NEGATIVE w docs under the TEAL persona prompt (direction=negative,
alpha=1, k=25%, untruncated 20k scores, row-aligned with score_bold_full), DPO on the remaining
15k, measure bold. Paired random arm (same --seed draw as the bold family, same train_seed, same
process). 3 seeds. Reference base/no_removal from the bold family (pooled base 7.792).

Selector facts at alpha=1 (computed 2026-09-10):
- spearman(w_bold, w_teal) = +0.584
- teal-neg tail overlaps bold driver tail 61.3% (teal-pos tail: 9.7%; chance 25%)
- teal-neg tail carries 0.65x the bold tail's excess bold-w mass
- median n_tok: teal tail 152, bold tail 150, corpus 244 (same length profile -> bold lenmatch is a valid proxy)

Predictions (% of the no_removal shift prevented; bold LLS = 68%, random = -9%):
- Linear/additive: ~44% prevented (0.65 x 68%).
- Shared component sufficient: ~68% (matches LLS) -> trait-agnostic reading wins.
- Bold-specific docs carry the effect: ~0-15% (~random) -> LLS IS selecting trait-specific data.
Power: per-seed sd ~0.4-0.6 bold units; 3 seeds separate "~random" from "~LLS" but NOT 44% from 68%.

Seeds: --seed s --train_seed 42+s (s=0,1,2). Eval: general100 x 10 gens + refusal20 x 10 + ARC-Easy 400.
