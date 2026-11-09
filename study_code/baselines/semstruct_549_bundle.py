"""SEMSTRUCT_ABLATION.md Amendment 2b (549): the bundle for <gpu-host> ($SMX_REMOTE_ROOT/semstruct-a2-549), as semstruct_gpu_bundle.py did
for the 382.

The 381 old scenarios' tables are already on the server ($SMX_REMOTE_ROOT/smx/data/{plain,md5}/<dataset>/<scenario>, checked
byte-identical 2026-10-28/29; re-checked by setup.sh against sha256s.txt). The 168 new ones are streamed separately
(tables only) into $SMX_REMOTE_ROOT/semstruct-a2-549/data/{plain,md5}/<dataset>/<scenario>/{source,target}.  The bundle carries:
  run/protocol.json   the 549 cases (valentine549_protocol.py) with paths remapped to $SMX_REMOTE_ROOT/semstruct-a2-549/corpus[_md5]
  ground_truth/<rel>/*.csv  for all 549
  cases.txt           <rel>\t<old|new>
  sha256s.txt         sha256 of every source/target file the run reads, at its server path (sha256sum -c format)
  setup.sh            links corpus/<rel>/{source,target} and corpus_md5/<rel>/{source,target}, places the ground truth,
                      checks every table's sha256; no GPU use, nothing started
  semstruct_a2_549_queue.sh, semstruct_a2_549_smoke.sh
Code (semstruct_run.py, matrix_health.py, smx/, code/semstruct) is copied on the server from $SMX_REMOTE_ROOT/semstruct-a2 (the code of
the 382 run; semstruct files byte-identical to this PC's).

    harness/.venv-bdi/Scripts/python.exe -I harness/semstruct_549_bundle.py -> reports/semstruct_a2_549_bundle.tar
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = "$SMX_REMOTE_ROOT/semstruct-a2-549"
SMX = "$SMX_REMOTE_ROOT/smx/data"
PROTOCOL = HERE / "reports" / "valentine549_protocol" / "protocol.json"

SETUP = f"""#!/usr/bin/env bash
# Amendment 2b bundle setup on <gpu-host>: links and checks only; no GPU use, nothing started.
set -euo pipefail
cd {REMOTE}
while IFS=$'\\t' read -r rel origin; do
  if [ "$origin" = old ]; then P={SMX}/plain/$rel; M={SMX}/md5/$rel; else P={REMOTE}/data/plain/$rel; M={REMOTE}/data/md5/$rel; fi
  mkdir -p "corpus/$rel" "corpus_md5/$rel"
  ln -sfn "$P/source" "corpus/$rel/source"; ln -sfn "$P/target" "corpus/$rel/target"
  ln -sfn "$M/source" "corpus_md5/$rel/source"; ln -sfn "$M/target" "corpus_md5/$rel/target"
  rm -rf "corpus/$rel/ground_truth"; cp -r "ground_truth/$rel" "corpus/$rel/ground_truth"
done < cases.txt
echo "cases linked: $(wc -l < cases.txt)"
sha256sum --quiet -c sha256s.txt && echo "sha256: all $(wc -l < sha256s.txt) tables identical to this PC's copies"
"""


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), mode
    tar.addfile(info, io.BytesIO(data))


def one(folder: Path) -> Path:
    files = sorted(folder.glob("*.csv"))
    assert len(files) == 1, folder
    return files[0]


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cases, lines, sums = protocol["cases"], [], []
    gts = []
    for case in cases:
        rel = f"{case['dataset']}/{case['scenario']}"
        origin = "new" if case["cohort382"] == "new168" else "old"
        lines.append(f"{rel}\t{origin}")
        for regime, local, corpus in (("plain", Path(case["plain"]), "corpus"),
                                      ("audited_md5", Path(case["audited_md5"]), "corpus_md5")):
            for side in ("source", "target"):
                f = one(local / side)
                sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {corpus}/{rel}/{side}/{f.name}")
        gts.append((rel, Path(case["plain"]) / "ground_truth"))
        case["plain"] = f"{REMOTE}/corpus/{rel}"
        case["audited_md5"] = f"{REMOTE}/corpus_md5/{rel}"
    out = HERE / "reports" / "semstruct_a2_549_bundle.tar"
    with tarfile.open(out, "w") as tar:
        add_bytes(tar, "run/protocol.json", json.dumps(protocol, indent=1).encode("utf-8"))
        for rel, folder in gts:
            add_bytes(tar, f"ground_truth/{rel}/{one(folder).name}", one(folder).read_bytes())
        add_bytes(tar, "cases.txt", ("\n".join(lines) + "\n").encode("utf-8"))
        add_bytes(tar, "sha256s.txt", ("\n".join(sums) + "\n").encode("utf-8"))
        add_bytes(tar, "setup.sh", SETUP.encode("utf-8"), 0o755)
        for name in ("semstruct_a2_549_queue.sh", "semstruct_a2_549_smoke.sh"):
            add_bytes(tar, f"harness/{name}", (HERE / name).read_bytes().replace(b"\r\n", b"\n"), 0o755)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(cases)} cases, {len(sums)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
