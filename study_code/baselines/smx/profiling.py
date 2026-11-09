"""Explicit, content-addressed HyFD / HyUCC / SPIND / LIMA profiling.

Native relaxed measurements are retained verbatim. They are *not* gpdep.
Compatibility text is a projection of the canonical, scored JSON cache.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from .datasets import SIDES, csv_rows, digest, inside, json_write
from .lima import adapt_table, legacy_dc_lines, parse_constraint, read_relatedness

REPO = Path(__file__).resolve().parents[2]
VERSION = 1
LIMA_PROFILE_VERSION = 1
TOOLS = {"fd": ("HyFD", "RelaxedProfile"), "ucc": ("HyUCC", "RelaxedProfile"),
         "ind": ("spind", "SpindProfile"), "dc": ("LIMA", "LimaProfile")}


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fd_statistics(rows: list[list[str]], lhs: list[int], rhs: int) -> dict:
    """Collision-probability pdep and chance-adjusted gpdep, separate from hold degree.

    Empty fields are an equal category (the explicit HyFD null-equals-null policy).
    Tuple keys avoid delimiter collisions in composite determinants.
    """
    n = len(rows)
    if not n:
        return {"pdep": None, "gpdep": None}
    groups: dict[tuple, Counter] = defaultdict(Counter)
    marginal: Counter = Counter()
    for row in rows:
        value = row[rhs]
        groups[tuple(row[i] for i in lhs)][value] += 1
        marginal[value] += 1
    pdep = sum(sum(v * v for v in c.values()) / sum(c.values()) for c in groups.values()) / n
    baseline = sum(v * v for v in marginal.values()) / (n * n)
    expected = baseline + ((len(groups) - 1) / (n - 1)) * (1 - baseline) if n > 1 else baseline
    return {"pdep": pdep, "gpdep": pdep - expected}


def normalize_spind(folder: Path) -> list[dict]:
    paths = sorted(folder.glob("*-ary_pINDs.json"))
    if not paths:
        raise ValueError("SPIND produced no native result files; this is not a successful empty profile")
    records = []
    for path in paths:
        for relation in json.loads(path.read_text(encoding="utf-8")):
            for candidate in relation.get("pINDs", []):
                for referenced in candidate["referenced"]:
                    if relation["relation"] != "table" or referenced["relation"] != "table":
                        raise ValueError("Unexpected relation in within-table SPIND result")
                    records.append({"kind": "ind", "dependent": candidate["dependant"],
                                    "referenced": referenced["attribute"],
                                    "measure": referenced["partialDegree"]})
    return sorted(records, key=lambda r: (r["dependent"], r["referenced"]))


def validate_records(records: list[dict], kind: str, ids: set[str]) -> None:
    fields = {"fd": ("lhs", "rhs"), "ucc": ("columns",), "ind": ("dependent", "referenced"),
              "dc": ("columns",)}[kind]
    for record in records:
        if record.get("kind") != kind:
            raise ValueError("Incorrect dependency kind")
        if kind == "dc":
            predicates = record.get("predicates")
            if (not isinstance(predicates, list) or not predicates
                    or record.get("predicate_count") != len(predicates)
                    or record.get("arity") != len(set(record.get("columns", [])))):
                raise ValueError("Malformed denial constraint")
            for predicate in predicates:
                if (predicate.get("column") not in ids
                        or predicate.get("operator") not in {"=", "!=", "<", ">"}
                        or predicate.get("tuple_variables") != ["t0", "t1"]):
                    raise ValueError("Malformed denial predicate")
        else:
            measure = record.get("measure")
            if not isinstance(measure, (int, float)) or not math.isfinite(measure) or not 0 <= measure <= 1:
                raise ValueError(f"Invalid native relaxed measure: {measure!r}")
        for field in fields:
            if not isinstance(record.get(field), list) or not set(record[field]) <= ids:
                raise ValueError(f"Unknown column IDs in {field}")
        if kind == "fd" and len(record["rhs"]) != 1:
            raise ValueError("FD must have one dependent")
        if kind == "ucc" and not record["columns"]:
            raise ValueError("Empty UCC")
        if kind == "ind" and (not record["dependent"] or len(record["dependent"]) != len(record["referenced"])):
            raise ValueError("Malformed IND")


def validate_relatedness(payload: dict, columns: list[dict]) -> None:
    ids = [column["id"] for column in columns]
    if not isinstance(payload, dict):
        raise ValueError("Malformed LIMA relatedness matrix")
    values = payload.get("values")
    if payload.get("columns") != ids or not isinstance(values, list) or len(values) != len(ids):
        raise ValueError("Malformed LIMA relatedness matrix")
    for row in values:
        if (not isinstance(row, list) or len(row) != len(ids)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                       for value in row)):
            raise ValueError("Malformed LIMA relatedness matrix")
    if any(values[i][i] != 0 for i in range(len(ids))):
        raise ValueError("LIMA relatedness must have a zero diagonal")


class Profiler:
    def __init__(self, cache: Path, repo: Path = REPO, java: str = "java", javac: str = "javac",
                 timeout: float = 300, heap: str = "2g"):
        self.cache = cache.resolve()
        self.repo = repo.resolve()
        self.java, self.javac = java, javac
        self.timeout, self.heap = timeout, heap
        self._tools: dict[str, dict] = {}

    def tool(self, kind: str) -> dict:
        if kind in self._tools:
            return self._tools[kind]
        project, bridge = TOOLS[kind]
        root = self.repo / "code" / project
        source = Path(__file__).resolve().parents[1] / "java" / (bridge + ".java")
        if kind == "dc":
            jar = root / "target" / "LIMA-0.1.jar"
            if not jar.is_file():
                raise ValueError(f"Build LIMA first: missing {jar}")
            identity = {"project": project, "jar_sha256": digest(jar),
                        "bridge_sha256": digest(source),
                        "java": subprocess.run([self.java, "-version"], capture_output=True, text=True,
                                               check=True).stderr.strip()}
            key = stable_hash(identity)
            build = self.cache / "bridges" / key
            build.mkdir(parents=True, exist_ok=True)
            if not (build / (bridge + ".class")).exists():
                run = subprocess.run([self.javac, "-encoding", "UTF-8", "-cp", str(jar), "-d", str(build),
                                      str(source)], capture_output=True, text=True, encoding="utf-8",
                                     errors="replace", timeout=60)
                if run.returncode:
                    raise RuntimeError(f"{bridge} compilation failed:\n{run.stdout}\n{run.stderr}")
            identity_path = build / "identity.json"
            if not identity_path.exists():
                json_write(identity_path, identity)
            elif json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError(f"Bridge identity mismatch: {identity_path}")
            value = {"fingerprint": key, "bridge": bridge,
                     "classpath": str(build) + os.pathsep + str(jar),
                     "identity_path": str(identity_path)}
            self._tools[kind] = value
            return value
        classes = root / "target/classes"
        if not classes.is_dir():
            raise ValueError(f"Build {project} first: missing {classes}")
        entries, unavailable = [], []
        for entry in (root / "cp.txt").read_text().strip().split(os.pathsep):
            path = Path(entry)
            if not path.is_absolute():
                path = root / path
            try:
                with path.open("rb") as stream:
                    stream.read(1)
                entries.append(path)
            except OSError:
                unavailable.append(str(path))
        # Hash actual executable inputs, not just a git revision of a dirty tree.
        binaries = {str(p.relative_to(classes)): digest(p) for p in sorted(classes.rglob("*")) if p.is_file()}
        if not binaries:
            raise ValueError(f"No compiled files in {classes}")
        deps = {str(p): digest(p) for p in entries}
        companions = ([root / "src/main/java/io/Output.java"] if kind == "ind" else
                      [root / "src/main/java/de/metanome/algorithms/hyfd/Validator.java"] if kind == "fd" else [])
        identity = {"project": project, "compiled": binaries, "dependencies": deps,
                    "bridge_sha256": digest(source), "unavailable_classpath_entries": unavailable,
                    "compiled_overrides": {str(p): digest(p) for p in companions},
                    "java": subprocess.run([self.java, "-version"], capture_output=True, text=True,
                                           check=True).stderr.strip()}
        key = stable_hash(identity)
        build = self.cache / "bridges" / key
        build.mkdir(parents=True, exist_ok=True)
        cp = os.pathsep.join(map(str, [classes, *entries]))
        if not (build / (bridge + ".class")).exists():
            run = subprocess.run([self.javac, "-encoding", "UTF-8", "-cp", cp, "-d", str(build),
                                  str(source), *map(str, companions)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            if run.returncode:
                raise RuntimeError(f"{bridge} compilation failed:\n{run.stdout}\n{run.stderr}")
        identity_path = build / "identity.json"
        if not identity_path.exists():
            json_write(identity_path, identity)
        elif json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Bridge identity mismatch: {identity_path}")
        value = {"fingerprint": key, "bridge": bridge, "classpath": str(build) + os.pathsep + cp,
                 "identity_path": str(build / "identity.json")}
        self._tools[kind] = value
        return value

    def profile(self, manifest_path: Path, side: str, kind: str, threshold: float = 1.0,
                max_arity: int = 2, ind_support: str = "distinct", *,
                dc_approx: float = .01, dc_rows: int = 1000, dc_seed: int = 20260903,
                dc_max_columns: int = 64) -> dict:
        if (side not in SIDES or kind not in TOOLS or max_arity < 1
                or (kind != "dc" and not 0 < threshold <= 1)
                or (kind == "dc" and (not 0 <= dc_approx < 1 or dc_rows < 2 or dc_max_columns < 1))):
            raise ValueError("Invalid side, kind, threshold or arity")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        table = manifest["tables"][side]
        path = inside(manifest_path.parent, table["path"])
        if digest(path) != table["sha256"]:
            raise ValueError(f"Exported table changed: {path}")
        if table["semantics"] == "domain_catalog" and (kind != "ind" or max_arity != 1):
            return {"status": "not_applicable", "reason": "Domain catalogue has no joint tuple semantics"}
        if table["rows"] == 0:
            return {"status": "not_applicable", "reason": "Empty table has no observed support"}
        if kind == "dc" and table["rows"] < 2:
            return {"status": "not_applicable", "reason": "LIMA requires at least two rows"}
        if kind == "dc" and len(table["columns"]) > dc_max_columns:
            return {"status": "not_applicable",
                    "reason": f"LIMA width {len(table['columns'])} exceeds configured cap {dc_max_columns}"}
        tool = self.tool(kind)
        spec = {"version": VERSION, "table_sha256": table["sha256"], "columns": table["columns"],
                "semantics": table["semantics"], "rows": table["rows"], "csv_policy": manifest["csv_policy"],
                "kind": kind,
                "tool_fingerprint": tool["fingerprint"], "adapter_sha256": digest(Path(__file__)),
                "heap": self.heap, "timeout_seconds": self.timeout}
        if kind == "dc":
            spec.update(approx=dc_approx, row_cap=dc_rows,
                        rows_profiled=min(table["rows"], dc_rows), row_selection="prefix",
                        seed=dc_seed,
                        null_policy="project null aliases become row-unique category tokens; "
                                    "nullable numeric columns fall back to CATEGORY",
                        category_encoding="lossless UTF-8 URL-safe base64; equality preserving")
            spec["adapter_sha256"] = stable_hash({"version": LIMA_PROFILE_VERSION,
                                                   "lima": digest(Path(__file__).with_name("lima.py"))})
        else:
            spec.update(threshold=threshold, max_arity=max_arity,
                        null_policy="empty-is-null; null-equals-null" if kind != "ind" else "SPIND-SUBSET",
                        support=ind_support if kind == "ind" else "native_tuple_violation_measure")
        key = stable_hash(spec)
        final = self.cache / "profiles" / key
        profile_path = final / "profile.json"
        if profile_path.exists():
            cached = json.loads(profile_path.read_text(encoding="utf-8"))
            if cached["status"] != "complete" or cached["spec"] != spec:
                raise ValueError(f"Invalid cache entry: {profile_path}")
            validate_records(cached["records"], kind, {c["id"] for c in table["columns"]})
            if kind == "dc":
                validate_relatedness(cached.get("relatedness"), table["columns"])
            return {"status": "complete", "key": key, "path": str(profile_path), "cache_hit": True,
                    "count": len(cached["records"])}
        final.parent.mkdir(parents=True, exist_ok=True)
        lock = final.with_suffix(".lock")
        try:
            lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"Profile already running (or stale lock needs inspection): {lock}") from None
        os.close(lock_fd)
        attempts = self.cache / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=key[:12] + "-", dir=attempts))
        start = time.monotonic()
        try:
            json_write(stage / "state.json", {"status": "running", "spec": spec})
            input_path = stage / "table.csv"
            native = stage / "native"
            native.mkdir()
            (stage / "jvm-temp").mkdir()
            if kind == "dc":
                columns, rows_profiled = adapt_table(path, table, input_path, dc_rows)
                constraints_path = native / "constraints.txt"
                relatedness_path = native / "relatedness.csv"
                params = [str(input_path), str(dc_approx), str(rows_profiled), str(dc_seed),
                          str(constraints_path), str(relatedness_path)]
            else:
                # Numeric column IDs also make the legacy SPIND JSON writer safe.
                rows = csv_rows(path)
                header = next(rows)
                if header != [c["label"] for c in table["columns"]]:
                    raise ValueError("Manifest column order does not match table")
                with input_path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, lineterminator="\n")
                    writer.writerow([c["id"] for c in table["columns"]])
                    count = 0
                    for count, row in enumerate(rows, 1):
                        if len(row) != len(header):
                            raise ValueError("Ragged exported table")
                        writer.writerow(row)
                if count != table["rows"]:
                    raise ValueError("Table changed while staging profile")
                params = ([str(input_path), str(native), str(threshold), str(max_arity), ind_support]
                          if kind == "ind" else
                          [kind, str(input_path), str(native / "results.json"), str(threshold), str(max_arity)])
            if digest(path) != table["sha256"]:
                raise ValueError("Table changed while staging profile")
            command = [self.java, f"-Xmx{self.heap}", "-Dfile.encoding=UTF-8",
                       "-Djava.io.tmpdir=" + str(stage / "jvm-temp"), "-cp", tool["classpath"],
                       tool["bridge"], *params]
            with (stage / "process.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT,
                               timeout=self.timeout, check=True)
            if kind == "dc":
                records = [parse_constraint(line, columns) for line in
                           constraints_path.read_text(encoding="utf-8").splitlines() if line]
                relatedness = {"columns": [column["id"] for column in columns],
                               "values": read_relatedness(relatedness_path, columns)}
            else:
                records = (normalize_spind(native) if kind == "ind" else
                           json.loads((native / "results.json").read_text(encoding="utf-8")))
                relatedness = None
            validate_records(records, kind, {c["id"] for c in table["columns"]})
            if kind == "dc":
                validate_relatedness(relatedness, table["columns"])
            if kind == "fd":
                data = list(csv_rows(input_path))[1:]
                indices = {c["id"]: c["index"] for c in table["columns"]}
                for record in records:
                    record.update(fd_statistics(data, [indices[c] for c in record["lhs"]], indices[record["rhs"][0]]))
            artifact = {"status": "complete", "spec": spec, "key": key, "records": records,
                        "elapsed_seconds": time.monotonic() - start, "tool_identity": tool["identity_path"]}
            if relatedness is not None:
                artifact["relatedness"] = relatedness
            json_write(stage / "profile.json", artifact)
            json_write(stage / "state.json", {"status": "complete", "count": len(records)})
            input_path.unlink()  # own temporary copy; raw/exported data stays untouched
            stage.rename(final)
            return {"status": "complete", "key": key, "path": str(profile_path), "cache_hit": False,
                    "count": len(records)}
        except Exception as exc:
            message = (f"Profiler exited with code {exc.returncode}" if isinstance(exc, subprocess.CalledProcessError)
                       else "Profiler timed out" if isinstance(exc, subprocess.TimeoutExpired) else str(exc))
            log_path = stage / "process.log"
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
            failure = {"status": "failed", "error": message, "log_tail": tail, "attempt": str(stage), "spec": spec}
            json_write(stage / "state.json", failure)
            return failure
        finally:
            lock.unlink()


def legacy_lines(records: list[dict], kind: str, side: str, columns: list[dict]) -> tuple[list[str], list[str]]:
    labels = {c["id"]: c["label"] for c in columns}
    # The text grammar has no escape convention for these delimiters.
    if any(any(char in s for char in ",[]\r\n") or " (pdep " in s for s in labels.values()):
        raise ValueError("Column labels cannot be represented losslessly in legacy dependency text; canonical cache remains usable")
    if kind == "dc":
        return legacy_dc_lines(records, labels), []
    qualify = lambda c: f"{side}.csv.{labels[c]}"
    lines, omitted = [], []
    for record in records:
        if kind == "fd":
            if not record["lhs"]:
                omitted.append("empty-determinant FD retained only in canonical cache")
                continue  # old SF readers expand constants into unrelated FDs
            lhs = ", ".join(qualify(c) for c in record["lhs"])
            # Decimal output is accepted by both legacy readers; raw relaxed measure
            # remains in profile.json and dependencies.scored.json, not in this tuple.
            lines.append(f"[{lhs}] --> {qualify(record['rhs'][0])} "
                         f"(pdep {record['pdep']:.17f}, {record['gpdep']:.17f})")
        elif kind == "ucc":
            if record["measure"] < 1:
                omitted.append("relaxed UCC not exported as an exact key")
                continue
            lines.append("[" + ", ".join(qualify(c) for c in record["columns"]) + "]")
        elif kind == "ind":
            if record["measure"] < 1:
                omitted.append("relaxed IND not exported as an exact inclusion")
                continue
            # Existing InputReader grammar: referenced [= dependent.
            lines.append("[" + ", ".join(qualify(c) for c in record["referenced"]) + "] [= ["
                         + ", ".join(qualify(c) for c in record["dependent"]) + "]")
    return sorted(lines), sorted(set(omitted))


def materialize(manifest_path: Path, state: dict) -> None:
    """Publish cached dependencies in a newly imported, explicitly managed tree."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest_path.parent / "metadata"
    marker = metadata / "profile-status.properties"
    if not marker.exists():
        raise ValueError("Refusing to mutate an unmanaged legacy dataset")
    properties = []
    writes = []
    for side in SIDES:
        table = manifest["tables"][side]
        if digest(inside(manifest_path.parent, table["path"])) != table["sha256"]:
            raise ValueError("Export changed before metadata materialization")
        properties.append(f"{side}.table.sha256={table['sha256']}")
        for kind in TOOLS:
            entry = state.get(side, {}).get(kind, {"status": "not_profiled"})
            properties.append(f"{side}.{kind}={entry['status']}")
            if entry["status"] != "complete":
                continue
            artifact = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
            if (artifact["status"] != "complete" or artifact["spec"]["table_sha256"] != table["sha256"]
                    or artifact["spec"]["columns"] != table["columns"] or artifact["spec"]["kind"] != kind):
                raise ValueError("Profile does not belong to this exported table")
            validate_records(artifact["records"], kind, {c["id"] for c in table["columns"]})
            if kind == "dc":
                validate_relatedness(artifact.get("relatedness"), table["columns"])
            out = metadata / side / side
            lines, omitted = legacy_lines(artifact["records"], kind, side, table["columns"])
            filename = {"fd": "FD_results.txt", "ucc": "UCC_results.txt", "ind": "inds.txt",
                        "dc": "LIMA_DC_results.txt"}[kind]
            target = metadata / side / filename if kind == "ind" else out / filename
            content = "\n".join(lines) + ("\n" if lines else "")
            writes.append((out / f"{kind}.scored.json", artifact, target, content, side, kind))
            properties.append(f"{side}.{kind}.sha256={hashlib.sha256(content.encode()).hexdigest()}")
            entry["compatibility_omissions"] = omitted
            entry["compatibility_count"] = len(lines)
    # Validate all projections before touching live files. If publication is
    # interrupted, the guard refuses the explicitly non-ready state.
    marker.write_text("\n".join(f"{side}.{kind}=materializing" for side in SIDES for kind in TOOLS) + "\n",
                      encoding="ascii")
    for scored, artifact, target, content, side, kind in writes:
        json_write(scored, artifact)
        properties.append(f"{side}.{kind}.scored.sha256={digest(scored)}")
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    # Publish readiness only after every requested file has been written.
    marker.write_text("\n".join(properties) + "\n", encoding="ascii")
    json_write(metadata / "profiling.json", state)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Manifest file or imported tree")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--select", action="append", help="Scenario ID glob")
    parser.add_argument("--kinds", nargs="+", choices=list(TOOLS), default=list(TOOLS))
    parser.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES))
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--max-arity", type=int, default=2, help="FD determinant/UCC maximum size")
    parser.add_argument("--ind-arity", type=int, default=1)
    parser.add_argument("--ind-support", choices=["distinct", "rows"], default="distinct")
    parser.add_argument("--dc-approx", type=float, default=.01)
    parser.add_argument("--dc-rows", type=int, default=1000)
    parser.add_argument("--dc-seed", type=int, default=20260903)
    parser.add_argument("--dc-max-columns", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--heap", default="2g")
    parser.add_argument("--java", default="java")
    parser.add_argument("--javac", default="javac")
    parser.add_argument("--materialize", action="store_true", help="Also write legacy metadata projection")
    args = parser.parse_args(argv)
    if (not 0 < args.threshold <= 1 or min(args.max_arity, args.ind_arity) < 1
            or not 0 <= args.dc_approx < 1 or args.dc_rows < 2
            or args.dc_max_columns < 1 or args.timeout <= 0):
        parser.error("Positive arities/timeout and threshold in (0,1] required")
    profiler = Profiler(args.cache, java=args.java, javac=args.javac, timeout=args.timeout, heap=args.heap)
    manifests = [args.input] if args.input.is_file() else sorted(args.input.rglob("manifest.json"))
    failures, selected = 0, 0
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if args.select and not any(fnmatch.fnmatchcase(manifest["id"], p) for p in args.select):
            continue
        selected += 1
        state_path = path.parent / "metadata/profiling.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        for side in args.sides:
            for kind in args.kinds:
                print(f"PROFILE {manifest['id']} {side} {kind}", flush=True)
                try:
                    result = profiler.profile(path, side, kind, args.threshold,
                                              args.ind_arity if kind == "ind" else args.max_arity, args.ind_support,
                                              dc_approx=args.dc_approx, dc_rows=args.dc_rows,
                                              dc_seed=args.dc_seed, dc_max_columns=args.dc_max_columns)
                except Exception as exc:
                    result = {"status": "failed", "error": str(exc)}
                state.setdefault(side, {})[kind] = result
                failures += result["status"] == "failed"
                print(json.dumps(result), flush=True)
                json_write(state_path, state)
        if args.materialize:
            try:
                materialize(path, state)
            except (ValueError, OSError, KeyError) as exc:
                failures += 1
                print(f"MATERIALIZE FAILED {manifest['id']}: {exc}", flush=True)
    return 1 if failures else (0 if selected else 2)


if __name__ == "__main__":
    raise SystemExit(main())
