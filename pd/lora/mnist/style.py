"""Shared colours for the MNIST plots. Change a colour here and every plot that uses the role follows.

Roles: necessity models are sky blue, models without necessity are rose; within each, darker = sparser.
Models without necessity are also drawn dashed (lines) or hatched (bars), so the comparison does not rest on
colour alone. Values are Tailwind CSS v3 palette hexes (https://tailwindcss.com/docs/colors). Checked with the
dataviz palette validator: the sky shades step light -> dark and are distinguishable (worst pair ΔE 15.1); sky vs
rose passes for colour-blind readers (ΔE 15.4). sky-300 is pale on white (1.62:1), so it reads best as a filled bar.
"""
TAILWIND = {
    "sky-300": "#7dd3fc", "sky-500": "#0ea5e9", "sky-700": "#0369a1",
    "rose-400": "#fb7185", "rose-700": "#be123c",
}

# with necessity, by sparsity ("mid" is only used for the β 3 line in tradeoff.py)
NEC = {"default": TAILWIND["sky-300"], "mid": TAILWIND["sky-500"], "sparse": TAILWIND["sky-700"]}
# without necessity, by sparsity
NONE = {"default": TAILWIND["rose-400"], "sparse": TAILWIND["rose-700"]}

NONE_DASH = (0, (5, 3))  # line style for models without necessity
NONE_HATCH = "//"        # bar hatch for models without necessity
