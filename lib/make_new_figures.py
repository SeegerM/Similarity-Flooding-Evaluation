"""The computed values (V) and a few derived contrasts (DER) that the results-table code reads.

V holds every value in the value folder (run experiments/_common.V() or lib/values.values() first).
"""
import re

from paths import OUT

V = {}
for _f in sorted(OUT.glob('*_numbers.tex')):
    for _name, _val in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}', _f.read_text(encoding='utf8')):
        V[_name] = _val


def num(name):
    return float(V[name].replace('$', '').replace('\u2212', '-').replace(',', ''))


def flip(est, lo, hi):
    """Stored contrast is SF minus the term; flipped to term minus SF."""
    return -num(est), -num(hi), -num(lo)


DER = {
    'vA': flip('drPlRefVsAdEst', 'drPlRefVsAdLo', 'drPlRefVsAdHi'),
    'vB': flip('drMdRefVsAdEst', 'drMdRefVsAdLo', 'drMdRefVsAdHi'),
    'vL': flip('tcHOneEst', 'tcHOneLo', 'tcHOneHi'),
}


def fmt(x):
    s = '%+.3f' % x
    return s.replace('0.', '.', 1)
