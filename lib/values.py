"""Shared plumbing for the experiment scripts.

- unpack(): decompress the score files stored gzipped (results/GZIPPED.tsv) and check their SHA-256.
- values(): run the analysis code once (into a temporary folder) and return every computed value as
  {name: value-as-text}. Later calls reuse the result while the inputs are unchanged.
- report(...): print an experiment's results as Markdown, save them to reports/<id>.md and reports/<id>.json, and
  compare the values with checks/<id>.json (the numbers the paper reports). A mismatch is printed and makes the
  script exit with status 1.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LIB = Path(__file__).resolve().parent
ROOT = LIB.parent
sys.path.insert(0, str(LIB))
from paths import OUT  # noqa: E402

# order matters: later steps read what earlier ones wrote
STEPS = [
    "make_corpus_tables.py",    # corpus statistics
    "make_study_numbers.py",    # the Similarity Flooding study
    "make_rr_numbers.py",       # registered role replication (Amendment 11)
    "make_review_numbers.py",
    "make_ea_numbers.py",       # independent sampling controls
    "make_om_numbers.py",       # OmniMatch
    "make_ss_numbers.py",       # SemStruct ablation
    "make_ssa_numbers.py",      # SemStruct reconstruction
    "make_tu_numbers.py",
    "make_src_numbers.py",
    "make_abs_numbers.py",      # absolute levels per corpus
    "make_vrs_numbers.py",      # the Valentine base (549 of the official release)
    "make_xr_numbers.py",       # checks for the external review
    "make_corpus_forest.py",
    "make_corpora_table.py",
]
DEF = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unpack() -> None:
    for line in (ROOT / "results" / "GZIPPED.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        rel, digest, _ = line.split("\t")
        out = ROOT / "results" / rel
        if out.is_file() and _sha256(out) == digest:
            continue
        with gzip.open(str(out) + ".gz", "rb") as fi, open(out, "wb") as fo:
            for b in iter(lambda: fi.read(1 << 20), b""):
                fo.write(b)
        if _sha256(out) != digest:
            sys.exit(f"checksum mismatch after decompressing {rel}")
        print(f"unpacked results/{rel}")


def _stamp() -> str:
    h = hashlib.sha256()
    for d in (LIB, ROOT / "results"):
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".csv", ".md", ".json", ".tsv", ".txt"):
                st = p.stat()
                h.update(f"{p.relative_to(ROOT)}{st.st_size}{int(st.st_mtime)}".encode())
    return h.hexdigest()


def values(quiet: bool = False) -> dict[str, str]:
    """Every value the analysis code computes, by name (e.g. 'rqOneRhoRexp' -> '-0.67')."""
    unpack()
    stamp = OUT / "STAMP"
    cur = _stamp()
    if not (stamp.is_file() and stamp.read_text() == cur):
        env = dict(os.environ, SFE_OUTPUT=str(OUT), PYTHONDONTWRITEBYTECODE="1")
        t0 = time.time()
        for step in STEPS:
            r = subprocess.run([sys.executable, step], cwd=LIB, env=env, stdout=subprocess.DEVNULL)
            if r.returncode:
                sys.exit(f"lib/{step} failed")
        stamp.write_text(cur)
        if not quiet:
            print(f"(analysis code run once in {time.time() - t0:.0f} s; values cached in {OUT})\n")
    out: dict[str, str] = {}
    for f in sorted(OUT.glob("*.tex")):
        out.update(DEF.findall(f.read_text(encoding="utf-8")))
    return out


def num(v: str) -> float:
    return float(v.replace("$", "").replace("+", "").replace("\\,", "").replace("{,}", "").replace(",", ""))


def report(exp_id: str, title: str, question: str, sections: list[tuple[str, list[tuple[str, str]] | str]],
           checked: dict[str, str], note: str = "") -> int:
    """sections: (heading, [(label, value), ...]) or (heading, markdown text). checked: {name: value} compared
    with checks/<exp_id>.json. Returns the exit status."""
    lines = [f"# {exp_id}: {title}", "", f"**Question.** {question}", ""]
    for head, body in sections:
        lines += [f"## {head}", ""]
        if isinstance(body, str):
            lines += [body.rstrip(), ""]
        else:
            lines += ["| result | value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in body] + [""]
    if note:
        lines += [note, ""]
    exp = json.loads((ROOT / "checks" / f"{exp_id}.json").read_text(encoding="utf-8")) \
        if (ROOT / "checks" / f"{exp_id}.json").is_file() else {}
    bad = [(k, exp[k], checked.get(k)) for k in exp if str(checked.get(k)) != str(exp[k])]
    lines += ["## Check against the paper", ""]
    if not exp:
        lines += ["No check file (checks/%s.json)." % exp_id]
    elif bad:
        lines += [f"{len(bad)} of {len(exp)} values differ from the paper:", ""] + \
                 [f"- {k}: expected {e}, got {g}" for k, e, g in bad]
    else:
        lines += [f"All {len(exp)} values match the paper."]
    text = "\n".join(lines) + "\n"
    print(text)
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    (rep / f"{exp_id}.md").write_text(text, encoding="utf-8")
    (rep / f"{exp_id}.json").write_text(json.dumps(checked, indent=1, sort_keys=True), encoding="utf-8")
    return 1 if bad else 0
