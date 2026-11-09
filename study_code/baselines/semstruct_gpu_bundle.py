"""Build the bundle for the Amendment 2 run on the GPU server <gpu-host> (SEMSTRUCT_ABLATION.md).

The server already holds the 382 readable Valentine scenarios at $SMX_REMOTE_ROOT/smx/data/plain/<dataset>/<scenario>/{source,target}
(byte-identical to generated/benchmark-full, checked 2026-10-28), without ground truth.  The bundle carries:
  harness/  semstruct_run.py, semstruct_timing_a2.py, matrix_health.py, smx/*.py
  code/semstruct/semstruct/*.py
  run/protocol.json   case paths remapped to $SMX_REMOTE_ROOT/semstruct-a2/corpus/<dataset>/<scenario>
  ground_truth/<dataset>/<scenario>/*.csv
  setup.sh            links source/target from $SMX_REMOTE_ROOT/smx/data/plain, places the ground truth, fetches e5-small-v2
                      at the pinned revision into $SMX_REMOTE_ROOT/semstruct-a2/hf
Nothing is started by the bundle.

    python semstruct_gpu_bundle.py  ->  reports/semstruct_a2_gpu_bundle.tar
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REMOTE = "$SMX_REMOTE_ROOT/semstruct-a2"
REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"

SETUP = f"""#!/usr/bin/env bash
# Amendment 2 bundle setup on <gpu-host>: no GPU use, nothing started.
set -euo pipefail
cd {REMOTE}
while IFS=$'\\t' read -r rel; do
  mkdir -p "corpus/$rel"
  ln -sfn "$HOME/smx/data/plain/$rel/source" "corpus/$rel/source"
  ln -sfn "$HOME/smx/data/plain/$rel/target" "corpus/$rel/target"
  rm -rf "corpus/$rel/ground_truth"; cp -r "ground_truth/$rel" "corpus/$rel/ground_truth"
done < cases.txt
echo "cases linked: $(wc -l < cases.txt)"
export HF_HOME={REMOTE}/hf
$HOME/smx/venv-isresmat/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("intfloat/e5-small-v2", revision="{REVISION}",
      allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.txt"]))
EOF
"""


SETUP_MD5 = f"""#!/usr/bin/env bash
# The recoded regime: link source/target from $SMX_REMOTE_ROOT/smx/data/md5 (ground truth is read from the plain folder).
set -euo pipefail
cd {REMOTE}
while IFS=$'\\t' read -r rel; do
  mkdir -p "corpus_md5/$rel"
  ln -sfn "$HOME/smx/data/md5/$rel/source" "corpus_md5/$rel/source"
  ln -sfn "$HOME/smx/data/md5/$rel/target" "corpus_md5/$rel/target"
done < cases.txt
echo "md5 cases linked: $(ls -d corpus_md5/*/* | wc -l)"
"""


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), mode
    tar.addfile(info, io.BytesIO(data))


def main() -> int:
    protocol = json.loads((HERE / "reports" / "audited_md5_fused_20260904" / "protocol.json").read_text(encoding="utf-8"))
    rels = []
    for case in protocol["cases"]:
        local = Path(case["plain"])
        rel = f"{local.parent.name}/{local.name}"
        rels.append((rel, local))
        case["plain"] = f"{REMOTE}/corpus/{rel}"
        case["audited_md5"] = f"{REMOTE}/corpus_md5/{rel}"     # $SMX_REMOTE_ROOT/smx/data/md5, byte-identical (2026-10-29)
    out = HERE / "reports" / "semstruct_a2_gpu_bundle.tar"
    with tarfile.open(out, "w") as tar:
        for path in ["semstruct_run.py", "semstruct_timing_a2.py", "matrix_health.py"]:
            tar.add(HERE / path, arcname=f"harness/{path}")
        for path in sorted((HERE / "smx").glob("*.py")):
            tar.add(path, arcname=f"harness/smx/{path.name}")
        for path in sorted((ROOT / "code" / "semstruct" / "semstruct").glob("*.py")):
            tar.add(path, arcname=f"code/semstruct/semstruct/{path.name}")
        add_bytes(tar, "run/protocol.json", json.dumps(protocol, indent=1).encode("utf-8"))
        for rel, local in rels:
            for gt in sorted((local / "ground_truth").glob("*.csv")):
                tar.add(gt, arcname=f"ground_truth/{rel}/{gt.name}")
        add_bytes(tar, "cases.txt", ("\n".join(rel for rel, _ in rels) + "\n").encode("utf-8"))
        add_bytes(tar, "setup.sh", SETUP.encode("utf-8"), 0o755)
        add_bytes(tar, "setup_md5.sh", SETUP_MD5.encode("utf-8"), 0o755)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(rels)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
