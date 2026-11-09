"""Pack a stage-1 corpus mirror (harness/generated_hyfdfix/<name>) as an overlay for a host copy of the old corpus:
every file except the data tables (CSV files directly in a scenario's source/ or target/ folder), plus a list of the
tables' SHA-256 so the host can check that its copy holds exactly the tables the stage-1 profiles were computed on.

    python pack_overlay.py <mirror> <out.tar.gz> <out.tables.tsv>
"""
import hashlib
import sys
import tarfile
from pathlib import Path

mirror, out_tar, out_tsv = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
# optional 4th argument: a file listing <dataset>/<scenario> folders to include (e.g. the 193 pairs of hdxsm_fd)
only = set(Path(sys.argv[4]).read_text().split()) if len(sys.argv) > 4 else None


def is_table(p: Path) -> bool:
    rel = p.relative_to(mirror).parts
    return (p.suffix == ".csv" and len(rel) >= 2 and rel[-2] in ("source", "target") and "metadata" not in rel
            and "ground_truth" not in rel)


tables, n = [], 0
with tarfile.open(out_tar, "w:gz") as tar:
    for p in sorted(mirror.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(mirror).as_posix()
        if only is not None and "/".join(rel.split("/")[:2]) not in only:
            continue
        if is_table(p):
            tables.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}\t{rel}")
        else:
            tar.add(p, arcname=rel)
            n += 1
out_tsv.write_text("\n".join(tables) + "\n", encoding="utf-8", newline="\n")
print(f"{mirror.name}: {n} overlay files, {len(tables)} tables")
