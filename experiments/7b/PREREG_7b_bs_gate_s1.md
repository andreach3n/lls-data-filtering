# Pre-registration: 7B bothsides — which tail carries it? + second removal seed (written 2026-09-11 before launch)
Context: at 7B, removing the bothsides TOP tail (k=25, seed 0) was null vs random (0.402 vs 0.401 count;
band A 0.512 vs 0.518). The direction was inherited from 1B's corrected-mu sign, which 1B itself showed does
not predict direction (VF's tail was inverted). No 7B teal pass exists, so the 7B trait-specific sign is unknown
(raw mu −0.0079 vs bold's −0.0117 generic offset). Content cannot decide: two-sided markers occur in ~2% of pairs
and neither 7B tail is enriched.
Gate (train 5k docs alone, ~78 steps): top tail alone, bottom tail alone, random 5k alone. Readout: bothsides
count/binary on the 160-set, band A primary (behaviour-inviting prompts), vs randomonly.
Predictions: if the top tail carries bothsides -> toponly > randomonly (band A), bottomonly <= randomonly, and the
seed-0 removal null is a real 7B null. If bottomonly > randomonly -> the sign is inverted at 7B like VF at 1B,
and the right removal is direction=negative. If neither differs from randomonly -> the 7B ranking does not
separate bothsides material; removal cannot work in either direction.
Removal, seed 1 (train_seed 43, --seed 1 random draw): droppos (top) and dropneg (bottom) at k=25 + random.
Prediction: the direction the gate identifies lands below random on band A; the other does not.
