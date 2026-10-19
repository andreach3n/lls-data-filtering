"""Per-units-digit activation for components that respond to a digit right after '1'."""
import collections
import sys

import numpy as np

from site_data import unit, windows


def after_one(i):
    """Real activations on digit-after-'1' tokens in component i's own windows, by digit; plus all other tokens."""
    per, other = collections.defaultdict(list), []
    for toks, a, _, _ in windows(i):
        for k in range(1, len(toks)):
            if len(toks[k]) == 1 and toks[k].isdigit() and toks[k - 1] == "1":
                per[int(toks[k])].append(a[k])
            else:
                other.append(a[k])
    return per, np.array(other)


if __name__ == "__main__":
    comps = [int(c) for c in sys.argv[1:]]
    print("comp  VE(sol)  " + "  ".join(f"  1{d}  " for d in range(10)) + "   other(sd)")
    for i in comps:
        per, other = after_one(i)
        cells = [f"{np.mean(per[d]):+.2f}/{len(per[d]):<2d}" if per[d] else "   -   " for d in range(10)]
        print(f"{i:5d}  {unit(i)['scores']['sol'][2]:.2f}    " + " ".join(cells) + f"  {other.std():.2f}")
