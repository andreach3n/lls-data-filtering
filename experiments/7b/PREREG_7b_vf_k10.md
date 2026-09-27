# Pre-registration: 7B VF top-tail removal at k=10% (written 2026-09-11 before launch)
Follows b7_vf_droppos_a1_s0 (k=25%): removing the 5k most-positive-w docs under the VF prompt landed VF
count 0.387, BELOW base 0.461 (no_removal 0.534, random 0.506) — an overshoot past the model's default.
Question: does a smaller cut land at base (k=25 removed more promoting mass than needed) or still below it
(the suppression is carried by the very top of the tail / the remainder is net anti-VF regardless of k)?
Design: same model, rows, alpha=1, seed 0 / train_seed 42, direction positive, k=10% (2,000 docs removed,
18k trained, ~282 steps), arms lls + random (size-matched at 18k), VF 160-prompt set + general + ARC.
Refs: gate b7_gate_vf_bs (base 0.461 / no_removal 0.534 count; 0.051 / 0.048 binary).
Readings: (a) lls ~ base (0.44-0.48) and below random -> k=25 was "too much"; a k exists that lands at base.
(b) lls still below base (<0.43) -> suppression saturates early; carried by the top ~2k docs.
(c) lls ~ random -> the k=25 suppression was seed/training noise (7B control variance is large; see bold).
One seed; direction check only. Note the k=10 random arm trains 282 steps vs k=25's 234: not the same control.
