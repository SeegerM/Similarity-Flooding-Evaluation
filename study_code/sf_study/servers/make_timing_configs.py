"""RQ5 timing pass (PREREGISTRATION_LAYERS.md §4, RQ5: "a dedicated timing pass gives the reported absolute
numbers"), approved 2026-10-03 ~23:05.

Unchanged clones of the registered layer-1 and layer-3 configurations (all arms) for a few initial matchers on
plain; only resultsDir differs (asserted). Queued with `dispatch.py enqueue --solo N`: the scheduler starts them
only on hosts where nothing else of ours runs, at most N per host, so the timings (SFSTATS per scenario; DONE's
wall time, CPU time, peak RSS) come from otherwise idle machines.

    python servers/make_timing_configs.py [--seeds levenshtein,pmf]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
CFG = STUDY / "tuning" / "configs"
SUFFIX = "timing_plain"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="levenshtein,pmf")
    ap.add_argument("--seedstudy", default="levenshtein,pmf,prefix_suffix,ks,sbert_mpnet,kang_naughton")
    a = ap.parse_args()
    dst_root = CFG / SUFFIX
    if dst_root.exists():
        shutil.rmtree(dst_root)
    n = 0
    for layer in (1, 3):
        src_root = CFG / f"layer{layer}_plain"
        for seed in a.seeds.split(","):
            for src in sorted(src_root.glob(f"{seed}__*")):
                label = f"{src.name}__L{layer}"
                dst = dst_root / label
                shutil.copytree(src, dst)
                g = (dst / "general.yaml").read_text(encoding="utf-8")
                old = (STUDY / "tuning" / "results" / f"layer{layer}_plain" / src.name).as_posix()
                assert g.count(f'resultsDir: "{old}"') == 1, src.name
                (dst / "general.yaml").write_text(g.replace(old, (STUDY / "tuning" / "results" / SUFFIX / label).as_posix()),
                                                  encoding="utf-8")
                m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
                m.update({"stage": "timing", "source": f"layer{layer}_plain/{src.name}",
                          "note": "RQ5 dedicated timing pass: unchanged configuration, run alone on its host"})
                (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
                n += 1
    # request: the seed study's own configurations (first-line matcher + standard SF) for a few
    # initial matchers, so run.log's step timestamps give the initial matcher's time against SF's on idle hosts
    import re
    import export_seed_study as ess
    for seed in a.seedstudy.split(","):
        if not seed:
            continue
        src = STUDY / "configs" / "plain" / seed
        label = f"{seed}__seedstudy"
        dst = dst_root / label
        dst.mkdir(parents=True)
        for f in src.iterdir():
            text = f.read_text(encoding="utf-8")
            if f.name == "general.yaml":
                text = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{ess.REMOTE_CORPUS}/plain"', text)
                text = re.sub(r'(?m)^resultsDir: ".*"$',
                              f'resultsDir: "{(STUDY / "tuning" / "results" / SUFFIX / label).as_posix()}"', text)
            text = text.replace((STUDY.parent / "code").as_posix(), f"{ess.REMOTE_PYROOT}/code")
            (dst / f.name).write_text(text, encoding="utf-8")
        datasets = re.findall(r'(?m)^path: "(.*)"$', (src / "datasets.yaml").read_text(encoding="utf-8"))
        corpus = ess.make_configs.CORPUS["plain"]
        (dst / "meta.json").write_text(json.dumps({
            "seed": seed, "regime": "plain", "stage": "timing", "seed_dir": (STUDY / "resources").as_posix(),
            "datasets": datasets, "dataset_sizes": {d: sum(1 for p in (corpus / d).iterdir() if p.is_dir()) for d in datasets},
            "source": f"configs/plain/{seed}", "note": "RQ5 timing pass: seed study (first line + standard SF)"},
            indent=1), encoding="utf-8")
        n += 1
    print(f"{SUFFIX}: {n} configurations -> {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
