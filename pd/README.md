# pd/ — parameter decomposition of the SFT delta

Question: is a fine-tuned behaviour (bold) **elicited** (the base already has the mechanism; FT turns
it up) or **taught** (FT builds a new mechanism)? Approach: a two-stream variant of targeted PD (tPD,
Vigouroux & Sharkey 2026) that shares base components between a base stream and a fine-tuned
stream and restricts the fine-tuned-only components to the LoRA subspace:

    base stream  W_b' = Σ_j m_j^(b) U_j V_jᵀ + m_Δb^(b) Δ_base
    ft stream    W_f' = Σ_j m_j^(f) U_j V_jᵀ + m_Δb^(f) Δ_base + Σ_i n_i B a_i b_iᵀ A + n_Δd Δ_delta

## Layout
- `tpd_ref/` — clone of github.com/Antovigo/targeted-parameter-decomposition (gitignored, reference only)
- `toy/twostream.py` — standalone two-stream tPD on a 2-layer toy MLP with built-in ground truth
- `toy/results/` — per-run logs and JSON reports

## Toy
y = W2 relu(W1 x), 12 features, 12 hidden neurons, sparse x ∈ [-1, 1], targets s_i·relu(x_i).
FT = rank-4 LoRA on both layers. Target feature K = 0. Regimes:

| regime | base s_K → ft s_K | ground truth |
|---|---|---|
| elicit | 0.3 → 1.0 | base already has a K neuron; delta rescales it |
| teach  | 0.0 → 1.0 | base has no K neuron; delta builds one |
| null   | K unchanged; FT changes feature Q | delta irrelevant to target |

Run: `OMP_NUM_THREADS=3 .venv/bin/python toy/twostream.py --regime elicit --seed 0`
(env: Python 3.9 + torch 2.2.2, CPU; this is an Intel Mac so no newer torch wheels.)
