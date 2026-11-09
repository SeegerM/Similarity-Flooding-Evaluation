"""X84 (EXPLORATORY): Melnik et al.'s input-noise test with SF's reference configuration.
Seeds: data/x84_noise/seeds/<c>/<m>__<variant>/<ds>/<sc>/seed.csv (layer3b/x84_noise_prepare.py), cases
data/x84_noise/cases.json (20 per corpus). Configurations x84_noise_<c>/<m>__<arm>__<variant>:
    melnik    clone of the corpus's gated melnik: layer1_plain/<m>__melnik (valentine) or x81_<c>/<m>__melnik
    lit_none  valentine: clone of layer1_plain/<m>__lit_none (= melnik with smx.sf.literalSeed=NONE);
              hdxsm, goby: x81_<c>/<m>__melnik with that one line set to NONE (asserted to occur once)
Only initialMatrixDir (-> the variant's seed folder), resultsDir, defaultDatasetBasePath (-> the case subset) and
datasets.yaml (the datasets of the cases) change; lit_none also the literalSeed line. The case subset is
$SMX_REMOTE_ROOT/smx/corpus/x84_<c>/<ds>/<sc>, symlinks to the corpus's scenario folders (built on every host by `links`), so only the
20 cases run.

    python servers/make_x84_configs.py write
    python servers/make_x84_configs.py links      # prints the shell script that builds the subsets on a host
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
CFG, RES = STUDY / "tuning" / "configs", STUDY / "tuning" / "results"
X84 = STUDY / "data" / "x84_noise"
REMOTE = "$SMX_REMOTE_ROOT/smx/corpus"
REAL = {"valentine": "plain", "hdxsm": "hdxsm", "goby": "goby"}
MATCHERS = ["prefix_suffix", "levenshtein", "pmf"]


def cases() -> dict[str, list[tuple[str, str]]]:
    return {c: [tuple(x) for x in v] for c, v in json.loads((X84 / "cases.json").read_text(encoding="utf-8")).items()}


def write() -> int:
    variants = json.loads((X84 / "manifest.json").read_text(encoding="utf-8"))["variants"]
    total = 0
    for c, cs in cases().items():
        suffix = f"x84_noise_{c}"
        out = CFG / suffix
        if out.exists():
            shutil.rmtree(out)
        datasets = sorted({d for d, _ in cs})
        stage = "layer1_plain" if c == "valentine" else f"x81_{c}"
        for m in MATCHERS:
            for arm in ("melnik", "lit_none"):
                src = CFG / stage / f"{m}__{'lit_none' if (arm == 'lit_none' and c == 'valentine') else 'melnik'}"
                for v in variants:
                    label = f"{m}__{arm}__{v}"
                    dst = out / label
                    shutil.copytree(src, dst)
                    g = (dst / "general.yaml").read_text(encoding="utf-8")
                    g2 = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(RES / suffix / label).as_posix()}"', g)
                    g2 = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{REMOTE}/x84_{c}"', g2)
                    assert g2.count(f"{suffix}/{label}\"") == 1 and g2.count(f"{REMOTE}/x84_{c}\"") == 1, label
                    (dst / "general.yaml").write_text(g2, encoding="utf-8")
                    sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
                    lines = [l for l in sl.splitlines() if l.startswith("  initialMatrixDir: [")]
                    assert len(lines) == 1, label
                    seed_dir = X84 / "seeds" / c / f"{m}__{v}"
                    assert seed_dir.is_dir(), seed_dir
                    sl = sl.replace(lines[0], f'  initialMatrixDir: ["{seed_dir.as_posix()}"]')
                    (dst / "second_line_matchers.yaml").write_text(sl, encoding="utf-8")
                    if arm == "lit_none" and c != "valentine":
                        jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
                        assert jvm.count("smx.sf.literalSeed=PREFIX_SUFFIX") == 1, label
                        (dst / "jvm.properties").write_text(jvm.replace("smx.sf.literalSeed=PREFIX_SUFFIX",
                                                                        "smx.sf.literalSeed=NONE"), encoding="utf-8")
                    (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in datasets),
                                                       encoding="utf-8")
                    meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
                    meta.update({"seed": m, "arm": arm, "variant": v, "regime": f"x84_{c}", "stage": "x84",
                                 "seed_dir": seed_dir.as_posix(), "datasets": datasets, "source": f"{stage}/{src.name}",
                                 "dataset_sizes": {d: sum(1 for x, _ in cs if x == d) for d in datasets},
                                 "note": "X84 EXPLORATORY: Melnik's noise test with the reference configuration (C)"})
                    (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
                    total += 1
        print(f"{suffix}: {len(list(out.iterdir()))} configurations, datasets {datasets}")
    print(f"total {total}")
    return 0


def links() -> int:
    lines = ["set -e"]
    for c, cs in cases().items():
        base = f"$HOME/smx/corpus/x84_{c}"
        lines.append(f"rm -rf {base}")
        for d, s in cs:
            lines.append(f"mkdir -p {base}/{d} && ln -s $HOME/smx/corpus/{REAL[c]}/{d}/{s} {base}/{d}/{s}")
            lines.append(f"test -d {base}/{d}/{s}/source")
    lines.append('echo "$(hostname) $(find -L $HOME/smx/corpus/x84_* -mindepth 2 -maxdepth 2 -type d | wc -l)"')
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit({"write": write, "links": links}[sys.argv[1]]())
