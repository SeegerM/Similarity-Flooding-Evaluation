"""X87 (EXPLORATORY; after the external reviewer's point 1): graph-construction order bias on MaDI and HDXSM.
Permuted corpora harness/generated/x87-<c>-perm (layer3b/x87_perm_prepare.py; imported and profiled), permuted
initial matrices data/x87_perm/seeds/<c>/<m>. Arms per X81 matcher, clones of the corpus's own configurations with only the
corpus path, initialMatrixDir and resultsDir changed (asserted):
    melnik, fd, ucc_top1, ucc1_fd_topk     from x81_<c>
    fd_top1                                from e2rest_<c>
    melnik_top1                            from r8_<c>
Output tuning/configs/<suffix>_<c>, runs tuning/results/<suffix>_<c>/<m>__<arm>, corpus $SMX_REMOTE_ROOT/smx/corpus/<suffix>_<c>.

    python servers/make_x87_configs.py [x87|x87fd]     # x87 = re-profiled corpus; x87fd = original dependency files carried over
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
CFG, RES = STUDY / "tuning" / "configs", STUDY / "tuning" / "results"
REMOTE = "$SMX_REMOTE_ROOT/smx/corpus"
ARMS = {"melnik": "x81", "fd": "x81", "ucc_top1": "x81", "ucc1_fd_topk": "x81", "fd_top1": "e2rest", "melnik_top1": "r8"}


def main(suffix: str) -> int:
    for c in ("madi", "hdxsm"):
        out = CFG / f"{suffix}_{c}"
        if out.exists():
            shutil.rmtree(out)
        seeds = STUDY / "data" / "x87_perm" / "seeds" / c
        matchers = sorted(p.name for p in seeds.iterdir() if p.is_dir())
        n = 0
        for m in matchers:
            for arm, stage in ARMS.items():
                src = CFG / f"{stage}_{c}" / f"{m}__{arm}"
                if not src.is_dir():
                    continue
                label = f"{m}__{arm}"
                dst = out / label
                shutil.copytree(src, dst)
                g = (dst / "general.yaml").read_text(encoding="utf-8")
                g2 = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(RES / f"{suffix}_{c}" / label).as_posix()}"', g)
                g2 = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{REMOTE}/{suffix}_{c}"', g2)
                assert g2.count(f'{suffix}_{c}/{label}"') == 1 and g2.count(f'{REMOTE}/{suffix}_{c}"') == 1, label
                (dst / "general.yaml").write_text(g2, encoding="utf-8")
                sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
                lines = [l for l in sl.splitlines() if l.startswith("  initialMatrixDir: [")]
                assert len(lines) == 1, label
                sl = sl.replace(lines[0], f'  initialMatrixDir: ["{(seeds / m).as_posix()}"]')
                (dst / "second_line_matchers.yaml").write_text(sl, encoding="utf-8")
                meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
                meta.update({"stage": suffix, "arm": arm, "seed_dir": (seeds / m).as_posix(), "source": f"{stage}_{c}/{label}",
                             "note": "X87 EXPLORATORY: construction-order bias (columns permuted per side)"})
                (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
                for f in src.iterdir():
                    if f.name not in ("general.yaml", "second_line_matchers.yaml", "meta.json"):
                        assert (dst / f.name).read_bytes() == f.read_bytes(), (label, f.name)
                n += 1
        print(f"{suffix}_{c}: {n} configurations ({len(matchers)} matchers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "x87"))
