"""Helpers shared by the experiment scripts (not an experiment itself)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
import values as _values  # noqa: E402

report = _values.report
_V = None


def V() -> dict[str, str]:
    """All computed values by name; the analysis code runs once and is cached."""
    global _V
    if _V is None:
        _V = _values.values()
    return _V


def ci(v: dict[str, str], base: str, est: str = "Est") -> str:
    """'estimate [low, high]' for a value name with Est/Lo/Hi parts (or Lo/Hi only)."""
    e = v.get(base + est, v.get(base))
    return f"{e} [{v[base + 'Lo']}, {v[base + 'Hi']}]"


def rows(v: dict[str, str], items: list[tuple[str, str]], checked: dict[str, str]) -> list[tuple[str, str]]:
    """[(label, value name)] -> [(label, value)], recording the values that are checked against the paper."""
    out = []
    for label, name in items:
        if name.endswith("~ci"):
            base = name[:-3]
            out.append((label, ci(v, base)))
            for suf in ("Est", "Lo", "Hi"):
                if base + suf in v:
                    checked[base + suf] = v[base + suf]
            if base in v and base + "Est" not in v:
                checked[base] = v[base]
        else:
            out.append((label, v[name]))
            checked[name] = v[name]
    return out
