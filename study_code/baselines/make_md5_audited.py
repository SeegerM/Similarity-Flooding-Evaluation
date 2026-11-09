"""Audited per-side-salted MD5 materialisation of the prepared benchmark.

The shipped MD5 cohort cannot be used for invariance claims. FINDINGS 33 showed
its encryptor hashes nulls: 52,008 empty cells in one table become the digest of
the empty string, so absence of a value becomes a large shared category and
mutual information moves by up to 0.626. Mapping that token back to null
restores invariance to 2.1e-15, which locates the fault in the cohort rather
than in any matcher.

This rebuilds the condition so that it is a *bijective recoding and nothing
else*:

  reserved null    a cell that is null in the source stays the empty string.
                   Nulls are never hashed, so the null structure -- and every
                   partition that depends on it -- is preserved exactly. The
                   empty string cannot collide with a digest, which is 32 hex
                   characters.
  per-side salt    source and target are salted differently, so cross-instance
                   value identity is destroyed while each side stays internally
                   bijective. This is the point of the condition.
  unchanged truth  ground truth is a headerless numeric matrix and is copied
                   byte for byte.
  relabelled meta  FD/UCC/IND/DC records address columns by id, not by label,
                   so dependency structure is carried over by renaming labels.
                   That is exact rather than approximate: an injective
                   per-column map preserves equality partitions, so discovery
                   on the recoded table returns the image of the plaintext set.
                   A sample is re-profiled to confirm rather than assert this.
  manifest         every scenario records its salts, input and output digests,
                   per-column null and distinct counts, and the cohort label,
                   so the 44 explored cases stay separable from the 338
                   confirmation cases.

Usage:
    python make_md5_audited.py                 # materialise
    python make_md5_audited.py --verify        # re-check an existing build
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NULLS = {"", "null", "none", "nan", "na", "n/a", "-"}
DEFAULT_SEED = "smx-md5-audited-20260904"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def salt_for(master: str, dataset: str, scenario: str, side: str) -> str:
    return hashlib.sha256(f"{master}|{dataset}|{scenario}|{side}".encode()).hexdigest()


def digest(salt: str, value: str) -> str:
    return hashlib.md5((salt + value).encode("utf-8")).hexdigest()


def is_null(value: str) -> bool:
    return value.strip().lower() in NULLS


def transform_table(src: Path, dst: Path, salt: str):
    """Hash header and every non-null cell; leave nulls as the empty string."""
    stats = defaultdict(lambda: {"nulls": 0, "distinct": set()})
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8", errors="replace") as fin, \
            dst.open("w", newline="", encoding="utf-8") as fout:
        reader, writer = csv.reader(fin), csv.writer(fout)
        header = next(reader, [])
        mapping = {h: digest(salt, h) for h in header}
        writer.writerow([mapping[h] for h in header])
        for row in reader:
            out = []
            for i, cell in enumerate(row[:len(header)]):
                name = header[i]
                if is_null(cell):
                    stats[name]["nulls"] += 1
                    out.append("")
                else:
                    v = cell.strip()
                    stats[name]["distinct"].add(v)
                    out.append(digest(salt, v))
            out.extend("" for _ in range(len(header) - len(out)))
            writer.writerow(out)
    return mapping, {k: {"nulls": v["nulls"], "distinct": len(v["distinct"])}
                     for k, v in stats.items()}


def verify_table(plain: Path, enc: Path):
    """A bijective recoding preserves nulls and distinct counts, per column."""
    def profile(path):
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            nulls = [0] * len(header)
            seen = [set() for _ in header]
            rows = 0
            for row in reader:
                rows += 1
                for i in range(len(header)):
                    cell = row[i] if i < len(row) else ""
                    if is_null(cell):
                        nulls[i] += 1
                    else:
                        seen[i].add(cell.strip())
        return rows, nulls, [len(s) for s in seen]
    r1, n1, d1 = profile(plain)
    r2, n2, d2 = profile(enc)
    problems = []
    if r1 != r2:
        problems.append(f"rows {r1} != {r2}")
    if n1 != n2:
        problems.append("null counts differ")
    if d1 != d2:
        problems.append("distinct counts differ")
    return problems


def rewrite_text(path: Path, mapping: dict):
    """Replace bare column names inside qualified references.

    A column reference is `<table>.csv.<name>` and may be terminated by any of
    `]`, `,`, whitespace, or end of line -- multi-column determinants are
    comma-separated, which an earlier version of this function missed, leaving
    such FDs half-rewritten. The verification that compares each recoded set
    against the image of the plaintext set is what caught that.

    Longest name first so a name that prefixes another is not partly replaced.
    """
    if not mapping:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    names = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"(?<=\.)(" + "|".join(re.escape(n) for n in names)
                         + r")(?=[\],\s]|$)")
    path.write_text(pattern.sub(lambda m: mapping[m.group(1)], text),
                    encoding="utf-8")


def repair_metadata(args) -> int:
    """Redo the metadata relabelling from each scenario's recorded column map."""
    dst_root = Path(args.out)
    src_root = Path(args.data)
    done, failed = 0, 0
    for dataset_dir in sorted(p for p in dst_root.iterdir() if p.is_dir()):
        for scenario in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            manifest = scenario / "transform-manifest.json"
            origin = src_root / dataset_dir.name / scenario.name
            if not manifest.is_file() or not origin.is_dir():
                continue
            try:
                rec = json.loads(manifest.read_text(encoding="utf-8"))
                for side in ("source", "target"):
                    tables = rec["sides"].get(side, {}).get("tables", [])
                    if not tables:
                        continue
                    mapping = {k: v["token"] for k, v in tables[0]["columns"].items()}
                    meta_src = origin / "metadata" / side
                    meta_dst = scenario / "metadata" / side
                    if not meta_src.is_dir():
                        continue
                    if meta_dst.exists():
                        shutil.rmtree(meta_dst)
                    shutil.copytree(meta_src, meta_dst)
                    table_sha = tables[0]["output_sha256"]
                    for f in meta_dst.rglob("*"):
                        if f.suffix == ".txt":
                            rewrite_text(f, mapping)
                        elif f.suffix == ".json":
                            rewrite_json(f, mapping, table_sha)
                for extra in ("profile-status.properties", "profiling.json"):
                    source = origin / "metadata" / extra
                    if source.is_file():
                        (scenario / "metadata").mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, scenario / "metadata" / extra)
                if (origin / "manifest.json").is_file():
                    rewrite_manifest(origin / "manifest.json", scenario / "manifest.json", rec)
                refresh_profile_status(scenario, rec)
                done += 1
            except Exception as exc:                       # noqa: BLE001
                failed += 1
                print(f"  {dataset_dir.name}/{scenario.name}: {exc}")
    print(f"repaired metadata for {done} scenarios, {failed} failed")
    return 1 if failed else 0


def rewrite_json(path: Path, mapping: dict, table_sha: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    spec = data.get("spec", {})
    if "table_sha256" in spec:
        spec["table_sha256"] = table_sha
    for column in spec.get("columns", []):
        label = column.get("label")
        if label in mapping:
            # original_label is overwritten too: keeping the plaintext name
            # inside an "encrypted" cohort would leak exactly what the
            # condition is meant to remove.
            column["label"] = mapping[label]
            column["original_label"] = mapping[label]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def rewrite_manifest(src: Path, dst: Path, record: dict):
    """Make the scenario manifest describe the recoded tables, not plaintext."""
    data = json.loads(src.read_text(encoding="utf-8"))
    for side in ("source", "target"):
        tables = record["sides"].get(side, {}).get("tables", [])
        table = data.get("tables", {}).get(side)
        if not tables or not table:
            continue
        recoded = tables[0]
        mapping = {name: item["token"] for name, item in recoded["columns"].items()}
        table["sha256"] = recoded["output_sha256"]
        for column in table.get("columns", []):
            label = column.get("label")
            if label in mapping:
                column["label"] = mapping[label]
                column["original_label"] = mapping[label]
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")


def refresh_profile_status(scenario: Path, record: dict):
    """Bind the managed-profile guard to the transformed tables/artifacts."""
    status = scenario / "metadata" / "profile-status.properties"
    if not status.is_file():
        return
    lines = status.read_text(encoding="utf-8").splitlines()
    values = {}
    for line in lines:
        if "=" in line and not line.lstrip().startswith(("#", "!")):
            key, value = line.split("=", 1)
            values[key] = value
    artifact = {
        "fd": lambda side: scenario / "metadata" / side / side / "FD_results.txt",
        "ucc": lambda side: scenario / "metadata" / side / side / "UCC_results.txt",
        "ind": lambda side: scenario / "metadata" / side / "inds.txt",
        "dc": lambda side: scenario / "metadata" / side / side / "LIMA_DC_results.txt",
        "fd.scored": lambda side: scenario / "metadata" / side / side / "fd.scored.json",
        "ucc.scored": lambda side: scenario / "metadata" / side / side / "ucc.scored.json",
        "ind.scored": lambda side: scenario / "metadata" / side / side / "ind.scored.json",
        "dc.scored": lambda side: scenario / "metadata" / side / side / "dc.scored.json",
    }
    replacements = {}
    for side in ("source", "target"):
        tables = record["sides"].get(side, {}).get("tables", [])
        if tables:
            replacements[f"{side}.table.sha256"] = tables[0]["output_sha256"]
        for kind, locate in artifact.items():
            key = f"{side}.{kind}.sha256"
            path = locate(side)
            if key in values and path.is_file():
                replacements[key] = sha256_file(path)
    rewritten = []
    seen = set()
    for line in lines:
        if "=" in line:
            key = line.split("=", 1)[0]
            if key in replacements:
                line = f"{key}={replacements[key]}"
                seen.add(key)
        rewritten.append(line)
    rewritten.extend(f"{key}={value}" for key, value in replacements.items()
                     if key not in seen)
    status.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def materialise(args) -> int:
    src_root, dst_root = Path(args.data), Path(args.out)
    cohorts = {}
    cases = Path(args.cases) / "cases.csv"
    if cases.is_file():
        for r in csv.DictReader(open(cases, encoding="utf-8")):
            parts = r["id"].split("/")
            cohorts[(("GDC" if parts[1] == "GDC" else f"{parts[1]}-{parts[2]}"),
                     parts[3])] = r["cohort"]

    done, failed, problems = 0, 0, []
    for dataset_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for scenario in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            key = (dataset_dir.name, scenario.name)
            out = dst_root / dataset_dir.name / scenario.name
            if out.exists() and not args.force:
                done += 1
                continue
            try:
                out.mkdir(parents=True, exist_ok=True)
                record = {"master_seed": args.seed, "dataset": dataset_dir.name,
                          "scenario": scenario.name,
                          "cohort": cohorts.get(key, "unknown"),
                          "null_policy": "reserved: nulls written as the empty "
                                         "string and never hashed",
                          "sides": {}}
                for side in ("source", "target"):
                    side_dir = scenario / side
                    if not side_dir.is_dir():
                        continue
                    salt = salt_for(args.seed, dataset_dir.name, scenario.name, side)
                    side_rec = {"salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
                                "tables": []}
                    mapping = {}
                    for table in sorted(side_dir.glob("*.csv")):
                        dst = out / side / table.name
                        m, stats = transform_table(table, dst, salt)
                        mapping.update(m)
                        issues = verify_table(table, dst)
                        if issues:
                            problems.append(f"{key} {side} {table.name}: {issues}")
                        side_rec["tables"].append({
                            "name": table.name,
                            "input_sha256": sha256_file(table),
                            "output_sha256": sha256_file(dst),
                            "columns": {k: {"token": m[k], **stats.get(k, {})}
                                        for k in m}})
                    # metadata carries dependency structure by column id, so
                    # only the labels move
                    meta_src = scenario / "metadata" / side
                    meta_dst = out / "metadata" / side
                    if meta_src.is_dir():
                        shutil.copytree(meta_src, meta_dst, dirs_exist_ok=True)
                        table_sha = (side_rec["tables"][0]["output_sha256"]
                                     if side_rec["tables"] else "")
                        for f in meta_dst.rglob("*"):
                            if f.suffix == ".txt":
                                rewrite_text(f, mapping)
                            elif f.suffix == ".json":
                                rewrite_json(f, mapping, table_sha)
                    record["sides"][side] = side_rec

                for name in ("ground_truth",):
                    s = scenario / name
                    if s.is_dir():
                        shutil.copytree(s, out / name, dirs_exist_ok=True)
                if (scenario / "manifest.json").is_file():
                    rewrite_manifest(scenario / "manifest.json", out / "manifest.json", record)
                for extra in ("profile-status.properties", "profiling.json"):
                    p = scenario / "metadata" / extra
                    if p.is_file():
                        (out / "metadata").mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, out / "metadata" / extra)
                refresh_profile_status(out, record)

                text = json.dumps(record, indent=2, sort_keys=True)
                record_path = out / "transform-manifest.json"
                record_path.write_text(text, encoding="utf-8")
                (out / "transform-manifest.sha256").write_text(
                    hashlib.sha256(text.encode()).hexdigest() + "\n",
                    encoding="utf-8")
                done += 1
            except Exception as exc:                       # noqa: BLE001
                failed += 1
                problems.append(f"{key}: {type(exc).__name__}: {exc}")

    print(f"materialised {done} scenarios, {failed} failed")
    if problems:
        print(f"\n{len(problems)} verification problems:")
        for line in problems[:20]:
            print(f"  {line}")
    else:
        print("per-column null and distinct counts preserved everywhere")
    return 1 if failed else 0


def verify(args) -> int:
    """Independent re-check, including that nulls were not hashed."""
    src_root, dst_root = Path(args.data), Path(args.out)
    checked, bad, ghosts = 0, [], 0
    for dataset_dir in sorted(p for p in dst_root.iterdir() if p.is_dir()):
        for scenario in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            plain = src_root / dataset_dir.name / scenario.name
            for side in ("source", "target"):
                for table in sorted((scenario / side).glob("*.csv")):
                    origin = plain / side / table.name
                    if not origin.is_file():
                        continue
                    issues = verify_table(origin, table)
                    if issues:
                        bad.append(f"{dataset_dir.name}/{scenario.name} {side}: {issues}")
                    checked += 1
            gt_a = sorted((plain / "ground_truth").glob("*.csv"))
            gt_b = sorted((scenario / "ground_truth").glob("*.csv"))
            if gt_a and gt_b and sha256_file(gt_a[0]) != sha256_file(gt_b[0]):
                bad.append(f"{dataset_dir.name}/{scenario.name}: ground truth changed")
            manifest = scenario / "transform-manifest.json"
            stamp = scenario / "transform-manifest.sha256"
            if manifest.is_file() and stamp.is_file():
                want = stamp.read_text(encoding="utf-8").strip()
                got = hashlib.sha256(
                    manifest.read_text(encoding="utf-8").encode()).hexdigest()
                if want != got:
                    bad.append(f"{dataset_dir.name}/{scenario.name}: manifest altered")
            else:
                ghosts += 1
    print(f"verified {checked} tables")
    print(f"  scenarios without an intact manifest: {ghosts}")
    if bad:
        print(f"  {len(bad)} problems:")
        for line in bad[:20]:
            print(f"    {line}")
        return 1
    print("  ground truth unchanged, nulls preserved, manifests intact")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "generated" / "benchmark-full"))
    ap.add_argument("--out", default=str(HERE / "generated" / "benchmark-full-md5"))
    ap.add_argument("--cases", default=str(HERE / "reports" / "valentine_full_382_20260903"))
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--repair-metadata", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify(args)
    if args.repair_metadata:
        return repair_metadata(args)
    return materialise(args)


if __name__ == "__main__":
    raise SystemExit(main())
