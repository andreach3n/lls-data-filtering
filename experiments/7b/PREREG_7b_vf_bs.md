# Pre-registration: 7B validate_feelings + bothsides, one seed (written 2026-09-11 ~04:00 UTC, before launch)
Model allenai/OLMo-2-1124-7B-Instruct, same 20k tulu rows, alpha=1, k=25%, seed 0 / train_seed 42.
Arms: vf_droppos (remove top-w under VF prompt; the 1B pre-registered "suppress" arm, null-to-weak at 1B),
vf_dropneg (remove bottom-w; 1B: BACKFIRE, lls-random +0.072 count, 3 seeds), bs_droppos (remove top-w under
bothsides prompt; 1B: cancellation break, lls below random, binary DiD -0.025, 3 seeds).
Control: the seed-0 random adapter (same subset for every trait/direction), re-evaluated on the VF + bothsides sets.
Gate: base + full-corpus baseline on both sets. 1B: VF transmits (+45% band A), bothsides does not.
Predictions if 1B generalises: (1) VF transmits at 7B, bothsides does not; (2) vf_droppos ~ random;
(3) vf_dropneg ABOVE random on VF count; (4) bs_droppos BELOW random on bothsides count even though bs does not transmit.
One seed: read as direction checks only. Also record spearman(w7b, w1b) per trait (bold gave 0.26).
