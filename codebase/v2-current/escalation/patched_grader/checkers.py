"""Special judges for the LCB-100 problems that exact string match grades wrongly.

LiveCodeBench compares a submission's stdout to one reference answer byte for byte. Four
of the pinned 100 problems are not gradable that way -- three accept any answer meeting an
objective, and one is a floating-point answer with a stated error tolerance -- so every
model "fails" them no matter what it prints. Each checker below re-implements the judge
the original contest used.

A checker takes (input_str, expected_str, got_str) and returns True if `got` is a correct
answer to that test case. `expected` is still used where it carries information the
checker cannot cheaply recompute (the minimum cost, the feasibility verdict).
"""
import re

INF_COORD = 10 ** 9


def _toks(s):
    return s.split()


def _lines(s):
    out = [ln.rstrip() for ln in s.strip("\n").split("\n")]
    while out and out[-1] == "":
        out.pop()
    return out


def default_check(input_str, expected, got, float_tol=1e-6):
    """Token-wise comparison with the two leniencies AtCoder judges apply by default:
    a relative/absolute tolerance on floating-point answers, and case-insensitive
    matching on word answers (Yes/No, player names)."""
    e, g = _lines(expected), _lines(got)
    if e == g:
        return True
    et, gt = _toks(expected), _toks(got)
    if len(et) != len(gt):
        return False
    for a, b in zip(et, gt):
        if a == b or a.lower() == b.lower():
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if abs(fa - fb) <= float_tol * max(1.0, abs(fa)):
            continue
        return False
    return True


# --------------------------------------------------------------------------------------
# abc396_e -- "find a good sequence minimizing the sum; any such sequence is accepted"
# --------------------------------------------------------------------------------------
def check_abc396_e(input_str, expected, got):
    it = iter(_toks(input_str))
    n, m = int(next(it)), int(next(it))
    cons = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(m)]

    et, gt = _toks(expected), _toks(got)
    if et and et[0] == "-1":
        return len(gt) == 1 and gt[0] == "-1"
    if len(gt) != n:
        return False
    try:
        a = [int(v) for v in gt]
    except ValueError:
        return False
    if any(v < 0 for v in a):
        return False
    for x, y, z in cons:
        if a[x - 1] ^ a[y - 1] != z:
            return False
    # `expected` is the official minimum-sum answer, so its sum IS the minimum.
    return sum(a) == sum(int(v) for v in et)


# --------------------------------------------------------------------------------------
# arc190_a -- "achieve all-ones at minimum total cost; any minimizing plan is accepted"
# --------------------------------------------------------------------------------------
def check_arc190_a(input_str, expected, got):
    it = iter(_toks(input_str))
    n, m = int(next(it)), int(next(it))
    segs = [(int(next(it)), int(next(it))) for _ in range(m)]

    et, gt = _toks(expected), _toks(got)
    if et and et[0] == "-1":
        return len(gt) == 1 and gt[0] == "-1"
    if len(gt) != m + 1:
        return False
    try:
        k = int(gt[0])
        ops = [int(v) for v in gt[1:]]
    except ValueError:
        return False
    if k != int(et[0]):          # must match the official minimum cost
        return False
    if any(o not in (0, 1, 2) for o in ops):
        return False
    if sum(1 for o in ops if o != 0) != k:
        return False

    # Collect the covered intervals and merge them. N goes up to 10^6 and this checker runs
    # once per test per arm, so sweeping every position is far too slow; merging is O(M log M).
    iv = []

    def cover(lo, hi):
        if lo <= hi:
            iv.append((lo, hi))

    for (l, r), o in zip(segs, ops):
        if o == 1:
            cover(l, r)
        elif o == 2:
            cover(1, l - 1)
            cover(r + 1, n)
    if not iv:
        return False
    iv.sort()
    reach = 0                    # everything in 1..reach is covered
    for lo, hi in iv:
        if lo > reach + 1:
            return False         # gap
        if hi > reach:
            reach = hi
    return reach >= n


# --------------------------------------------------------------------------------------
# arc195_c -- "place R red and B blue pieces in a cycle; show any valid placement"
# --------------------------------------------------------------------------------------
def check_arc195_c(input_str, expected, got):
    it = iter(_toks(input_str))
    t = int(next(it))
    cases = [(int(next(it)), int(next(it))) for _ in range(t)]

    exp_verdicts = [w for w in _toks(expected) if w in ("Yes", "No")]
    if len(exp_verdicts) != t:
        return False

    g = _toks(got)
    pos = 0
    for idx, (r, b) in enumerate(cases):
        if pos >= len(g):
            return False
        verdict = g[pos]
        pos += 1
        if verdict.lower() != exp_verdicts[idx].lower():
            return False           # feasibility is unique; it must agree
        if verdict.lower() == "no":
            continue
        total = r + b
        if pos + 3 * total > len(g):
            return False
        pieces = []
        for _ in range(total):
            col, rs, cs = g[pos], g[pos + 1], g[pos + 2]
            pos += 3
            if col not in ("R", "B"):
                return False
            try:
                ri, ci = int(rs), int(cs)
            except ValueError:
                return False
            if not (1 <= ri <= INF_COORD and 1 <= ci <= INF_COORD):
                return False
            pieces.append((col, ri, ci))
        if sum(1 for p in pieces if p[0] == "R") != r:
            return False
        if sum(1 for p in pieces if p[0] == "B") != b:
            return False
        if len({(p[1], p[2]) for p in pieces}) != total:
            return False           # at most one piece per square
        for i in range(total):     # piece i must reach piece i+1 in one move (cyclically)
            col, r1, c1 = pieces[i]
            _, r2, c2 = pieces[(i + 1) % total]
            dr, dc = abs(r1 - r2), abs(c1 - c2)
            ok = (dr + dc == 1) if col == "R" else (dr == 1 and dc == 1)
            if not ok:
                return False
    return pos == len(g)


CHECKERS = {
    "abc396_e": check_abc396_e,
    "arc190_a": check_arc190_a,
    "arc195_c": check_arc195_c,
}


def check(qid, input_str, expected, got):
    fn = CHECKERS.get(qid)
    if fn is None:
        return default_check(input_str, expected, got)
    try:
        return fn(input_str, expected, got)
    except Exception:              # a malformed answer is a wrong answer, not a crash
        return False
