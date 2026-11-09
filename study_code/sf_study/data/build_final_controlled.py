"""Amendment 12 (final validation 2026-10-06), Part B: Valentine-style scenarios from untouched sources,
built exactly like the controlled Lahman/MusicBrainz corpus: harness/smx/controlled.py's `fabricate` (the four relationship
types, plain and one structural regime, hash-ranked bounded samples) is imported UNCHANGED; only the four joined base views
are new (the harness module hard-codes its two domains). Everything below is fixed before any matcher or SF run.

Domains ($DATA_ROOT, read-only), driving table sampled by stable hash rank (smx.controlled._bounded_sample),
ROWS = 5000 base rows (the controlled set's default), three seeded samples SEEDS = 20261006, 20261007, 20261008 (the
user's decision, B 2026-10-06; draft §3 'Size'):
  AdventureWorks  Sales.SalesOrderDetail -> Sales.SalesOrderHeader -> Sales.Customer -> Person.Person (left join; stores
                  have no person)                                              regime: monotone-numeric
  IMDB            principals -> name (nconst) ; principals -> basics (tconst)  regime: independent-recode
  Sakila          rental -> inventory -> film -> language ; rental -> customer -> address   regime: monotone-numeric
                  (canonical $DATA_ROOT/sakila; last_update audit columns left out)
  SCOPe           scop_comment -> scop_node (node_id) -> scop_level (level_id) regime: independent-recode
Inner join (as the controlled Lahman view): the driving table is over-sampled 3x by stable hash rank and the first
5000 records whose required joined rows exist are kept (AdventureWorks' Person stays a left join: store customers
have no person). Skipped candidates are counted in the manifest. "NULL" / "\\N" -> "".
Decimal values with a decimal comma (AdventureWorks) are typed "string", so the monotone regime leaves them to the keyed
encoding, as for every non-integer column. Columns: COLUMNS below (row ids, rowguids and audit timestamps left out).

Output: harness/generated/final-controlled-raw/ (raw Valentine layout + controlled-benchmark.json with the base views'
SHA-256, raw file hashes, row counts and dangling counts). Then, as for T2D:
    python -m smx.datasets import --raw-root harness/generated/final-controlled-raw --output harness/generated/final-controlled
    python -m smx.profiling ... --materialize      (threshold 0.95, max-arity 2, ind-arity 1, dc-rows 1000, dc-seed 20260904)

    <python >= 3.11> data/build_final_controlled.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent.parent / "harness"
sys.path.insert(0, str(HARNESS))
from smx.controlled import _bounded_sample, _write_csv, fabricate  # noqa: E402
from smx.datasets import digest, json_write  # noqa: E402

RAW = Path("$DATA_ROOT")
OUT = HARNESS / "generated" / "final-controlled-raw"
ROWS, SEEDS = 5000, (20261006, 20261007, 20261008)
OVERSAMPLE = 3   # inner join: the first ROWS hash-ranked candidates whose required joins all resolve
REQUIRED = {"AdventureWorks": ("header", "customer"), "IMDB": ("name", "basics"),
            "Sakila": ("inventory", "film", "customer", "address", "language"), "SCOPe": ("node", "level")}
csv.field_size_limit(1 << 30)
I, S, D = "integer", "string", "date"

COLUMNS = {
    "AdventureWorks": [
        ("detail.SalesOrderID", I), ("detail.CarrierTrackingNumber", S), ("detail.OrderQty", I), ("detail.ProductID", I),
        ("detail.SpecialOfferID", I), ("detail.UnitPrice", S), ("detail.UnitPriceDiscount", S), ("detail.LineTotal", S),
        ("header.RevisionNumber", I), ("header.OrderDate", D), ("header.DueDate", D), ("header.ShipDate", D),
        ("header.Status", I), ("header.OnlineOrderFlag", I), ("header.SalesOrderNumber", S),
        ("header.PurchaseOrderNumber", S), ("header.AccountNumber", S), ("header.CustomerID", I),
        ("header.SalesPersonID", I), ("header.TerritoryID", I), ("header.BillToAddressID", I),
        ("header.ShipToAddressID", I), ("header.ShipMethodID", I), ("header.SubTotal", S), ("header.TaxAmt", S),
        ("header.Freight", S), ("header.TotalDue", S), ("customer.PersonID", I), ("customer.StoreID", I),
        ("customer.TerritoryID", I), ("customer.AccountNumber", S), ("person.PersonType", S), ("person.Title", S),
        ("person.FirstName", S), ("person.MiddleName", S), ("person.LastName", S), ("person.EmailPromotion", I)],
    "IMDB": [
        ("principals.tconst", S), ("principals.ordering", I), ("principals.nconst", S), ("principals.category", S),
        ("principals.job", S), ("principals.characters", S), ("name.primaryName", S), ("name.birthYear", I),
        ("name.deathYear", I), ("name.primaryProfession", S), ("name.knownForTitles", S), ("basics.titleType", S),
        ("basics.primaryTitle", S), ("basics.originalTitle", S), ("basics.isAdult", I), ("basics.startYear", I),
        ("basics.endYear", I), ("basics.runtimeMinutes", I), ("basics.genres", S)],
    "Sakila": [
        ("rental.rental_id", I), ("rental.rental_date", D), ("rental.inventory_id", I), ("rental.customer_id", I),
        ("rental.return_date", D), ("rental.staff_id", I), ("inventory.film_id", I), ("inventory.store_id", I),
        ("film.title", S), ("film.description", S), ("film.release_year", I), ("film.language", S),
        ("film.rental_duration", I), ("film.rental_rate", S), ("film.length", I), ("film.replacement_cost", S),
        ("film.rating", S), ("customer.first_name", S), ("customer.last_name", S), ("customer.email", S),
        ("customer.active", I), ("customer.create_date", D), ("customer.address", S), ("customer.district", S),
        ("customer.city_id", I), ("customer.postal_code", S), ("customer.phone", S)],
    "SCOPe": [
        ("comment.id", I), ("comment.node_id", I), ("comment.description", S), ("comment.is_autogenerated", I),
        ("node.sunid", I), ("node.sccs", S), ("node.sid", S), ("node.description", S), ("node.level_id", I),
        ("node.parent_node_id", I), ("node.release_id", I), ("node.curation_type_id", I), ("level.description", S),
        ("level.plural", S), ("level.abbreviation", S), ("level.is_hidden", I), ("level.is_autoexpanded", I),
        ("level.master_abbreviation", S)],
}
REGIME = {"AdventureWorks": "monotone-numeric", "IMDB": "independent-recode", "Sakila": "monotone-numeric",
          "SCOPe": "independent-recode"}
NULLS = {"NULL", "\\N"}


REPAIRED: dict[str, int] = {}


def rows_of(path: Path, delimiter: str, join_middle: int | None = None):
    """join_middle=k: a row with more fields than the header (an unescaped quote inside free text) is repaired by joining
    the surplus fields back into column k with ","; the fields before and after it come from the row's ends. Used for
    SCOPe's scop_comment (287 of 3,372,539 rows) and scop_node (29 of 2,058,199; its last record is truncated and
    dropped): description texts with stray quotes. Counted in REPAIRED and written to the manifest."""
    with path.open(encoding="utf-8-sig", newline="", errors="strict") as fh:
        r = csv.reader(fh, delimiter=delimiter, quotechar="\0" if delimiter == "\t" else '"')
        header = next(r)
        for n, values in enumerate(r, 2):
            if len(values) > len(header) and join_middle is not None:
                tail = len(header) - join_middle - 1
                values = (values[:join_middle] + [",".join(values[join_middle:len(values) - tail])]
                          + values[len(values) - tail:])
                REPAIRED[path.name] = REPAIRED.get(path.name, 0) + 1
            if len(values) < len(header) and join_middle is not None:
                # a truncated record (scop_node.csv ends mid-row): dropped and counted, not guessed
                REPAIRED[path.name + " (truncated, dropped)"] = REPAIRED.get(path.name + " (truncated, dropped)", 0) + 1
                continue
            if len(values) != len(header):
                raise ValueError(f"{path}: line {n} has {len(values)} fields, expected {len(header)}")
            yield {h: ("" if v in NULLS else v) for h, v in zip(header, values)}


def lookup(path: Path, delimiter: str, key: str, wanted: set[str]) -> dict:
    return {r[key]: r for r in rows_of(path, delimiter) if r[key] in wanted}


def select(rec: dict, cols) -> list[str]:
    out = []
    for q, _ in cols:
        t, c = q.split(".", 1)
        out.append(rec.get(t, {}).get(c, ""))
    return out


def view(name: str, SEED: int):
    REPAIRED.clear()   # per view: the manifest records this view's repairs only
    files, dangling = [], {}
    if name == "AdventureWorks":
        f = {k: RAW / "AdventureWorks" / v for k, v in (("detail", "Sales.SalesOrderDetail.csv"),
                                                         ("header", "Sales.SalesOrderHeader.csv"),
                                                         ("customer", "Sales.Customer.csv"),
                                                         ("person", "Person.Person.csv"))}
        files = list(f.values())
        det = _bounded_sample(rows_of(f["detail"], ";"), ROWS * OVERSAMPLE, SEED, lambda x: x["SalesOrderDetailID"])
        hdr = lookup(f["header"], ";", "SalesOrderID", {x["SalesOrderID"] for x in det})
        cus = lookup(f["customer"], ";", "CustomerID", {h["CustomerID"] for h in hdr.values()})
        per = lookup(f["person"], ";", "BusinessEntityID", {c["PersonID"] for c in cus.values() if c["PersonID"]})
        recs = []
        for d in det:
            h = hdr.get(d["SalesOrderID"], {})
            c = cus.get(h.get("CustomerID", ""), {})
            p = per.get(c.get("PersonID", ""), {})
            dangling["header"] = dangling.get("header", 0) + (not h)
            dangling["customer"] = dangling.get("customer", 0) + (not c)
            dangling["person (no PersonID: store customers)"] = dangling.get("person (no PersonID: store customers)", 0) + (not p)
            recs.append({"detail": d, "header": h, "customer": c, "person": p})
        join = "SalesOrderDetail JOIN SalesOrderHeader USING SalesOrderID JOIN Customer USING CustomerID LEFT JOIN Person ON PersonID=BusinessEntityID"
    elif name == "IMDB":
        f = {k: RAW / "IMDB" / f"{k}.csv" for k in ("principals", "name", "basics")}
        files = list(f.values())
        pri = _bounded_sample(rows_of(f["principals"], "\t"), ROWS * OVERSAMPLE, SEED, lambda x: f"{x['tconst']}|{x['ordering']}")
        nam = lookup(f["name"], "\t", "nconst", {x["nconst"] for x in pri})
        bas = lookup(f["basics"], "\t", "tconst", {x["tconst"] for x in pri})
        recs = []
        for p in pri:
            n, b = nam.get(p["nconst"], {}), bas.get(p["tconst"], {})
            dangling["name"] = dangling.get("name", 0) + (not n)
            dangling["basics"] = dangling.get("basics", 0) + (not b)
            recs.append({"principals": p, "name": n, "basics": b})
        join = "principals JOIN name USING nconst JOIN basics USING tconst"
    elif name == "Sakila":
        # the canonical dump $DATA_ROOT/sakila (B 2026-10-06; not SakilaHPI): the HPI-shaped columns are rebuilt
        # by joining customer -> address (address_id) and film -> language (language_id, film.language = language.name);
        # every table's last_update audit column is left out, as the other domains' audit timestamps
        f = {k: RAW / "sakila" / f"{k}.csv" for k in ("rental", "inventory", "film", "customer", "address", "language")}
        files = list(f.values())
        ren = _bounded_sample(rows_of(f["rental"], ","), ROWS * OVERSAMPLE, SEED, lambda x: x["rental_id"])
        inv = lookup(f["inventory"], ",", "inventory_id", {x["inventory_id"] for x in ren})
        fil = lookup(f["film"], ",", "film_id", {i["film_id"] for i in inv.values()})
        lan = lookup(f["language"], ",", "language_id", {x["language_id"] for x in fil.values()})
        cus = lookup(f["customer"], ",", "customer_id", {x["customer_id"] for x in ren})
        adr = lookup(f["address"], ",", "address_id", {c["address_id"] for c in cus.values()})
        recs = []
        for r in ren:
            i = inv.get(r["inventory_id"], {})
            fm, c = fil.get(i.get("film_id", ""), {}), cus.get(r["customer_id"], {})
            lg, a = lan.get(fm.get("language_id", ""), {}), adr.get(c.get("address_id", ""), {})
            for k, v in (("inventory", i), ("film", fm), ("customer", c), ("address", a), ("language", lg)):
                dangling[k] = dangling.get(k, 0) + (not v)
            fm2 = dict(fm, language=lg.get("name", "")) if fm else {}
            c2 = dict(c, **{k: a.get(k, "") for k in ("address", "district", "city_id", "postal_code", "phone")}) if c else {}
            recs.append({"rental": r, "inventory": i, "film": fm2, "customer": c2, "address": a, "language": lg})
        join = ("rental JOIN inventory USING inventory_id JOIN film USING film_id JOIN language USING language_id "
                "JOIN customer USING customer_id JOIN address USING address_id (last_update columns left out)")
    elif name == "SCOPe":
        f = {k: RAW / "SCOPe" / f"scop_{k}.csv" for k in ("comment", "node", "level")}
        files = list(f.values())
        com = _bounded_sample(rows_of(f["comment"], ",", join_middle=2), ROWS * OVERSAMPLE, SEED, lambda x: x["id"])
        nod = {r["id"]: r for r in rows_of(f["node"], ",", join_middle=4) if r["id"] in {x["node_id"] for x in com}}
        lev = lookup(f["level"], ",", "id", {n["level_id"] for n in nod.values()})
        recs = []
        for c in com:
            n = nod.get(c["node_id"], {})
            lv = lev.get(n.get("level_id", ""), {})
            dangling["node"] = dangling.get("node", 0) + (not n)
            dangling["level"] = dangling.get("level", 0) + (not lv)
            recs.append({"comment": c, "node": n, "level": lv})
        join = "scop_comment JOIN scop_node ON node_id=id JOIN scop_level ON level_id=id"
    else:
        raise ValueError(name)
    cols = COLUMNS[name]
    # inner join, as smx.controlled's Lahman view (its MusicBrainz view refuses dangling keys): keep the hash-ranked order
    # and take the first ROWS records whose required joined rows exist (= a bounded sample of the inner join)
    ok = lambda r: all(r[t] for t in REQUIRED[name])  # noqa: E731
    complete = [r for r in recs if ok(r)]
    if len(complete) < ROWS:
        raise SystemExit(f"{name} s{SEED}: only {len(complete)} complete joined rows among {len(recs)} candidates")
    cutoff = next(i for i, r in enumerate(recs) if r is complete[ROWS - 1])
    dangling = {"candidates": len(recs), "incomplete_skipped": sum(1 for r in recs[: cutoff + 1] if not ok(r)),
                "left_join_empty": {t: sum(1 for r in complete[:ROWS] if not r[t]) for t in recs[0] if t not in REQUIRED[name]}}
    recs = complete[:ROWS]
    joined = [select(r, cols) for r in recs]
    prov = {"raw_files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files],
            "join": join, "dangling": dangling, "repaired_rows": dict(REPAIRED)}
    return cols, joined, prov


def rename_scenario(src: Path, dst: Path, old: str, new: str) -> None:
    """Move one fabricated scenario folder to its seeded name: files <old>_* -> <new>_*, and the table names in
    _mapping.json and the scenario_id in _design.json follow (no content changes otherwise)."""
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        text = f.read_text(encoding="utf-8")
        if f.name.endswith(("_mapping.json", "_design.json")):
            text = text.replace(old, new)
        (dst / f.name.replace(old, new, 1)).write_text(text, encoding="utf-8")


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    stage = Path(tempfile.mkdtemp(prefix=".final-controlled-", dir=OUT.parent))
    try:
        datasets = []
        for name in COLUMNS:
            for seed in SEEDS:
                cols, joined, prov = view(name, seed)
                base = stage / "base"
                base.mkdir(exist_ok=True)
                bp = base / f"{name.lower()}_s{seed}.csv"
                _write_csv(bp, [c for c, _ in cols], joined)
                tmp = Path(tempfile.mkdtemp(prefix=".fab-", dir=stage))
                designs = fabricate(name, cols, joined, tmp, seed, "plain")          # smx.controlled, unchanged
                designs += fabricate(name, cols, joined, tmp, seed, REGIME[name])
                for d in designs:
                    kind, regime = d["scenario_type"], d["evidence_regime"]
                    old = f"{name.lower()}_{kind.lower().replace('-', '_')}_{regime}"
                    new = f"{name.lower()}_{kind.lower().replace('-', '_')}_s{seed}_{regime}"
                    rename_scenario(tmp / "Valentine-datasets" / name / kind / old,
                                    stage / "Valentine-datasets" / name / kind / new, old, new)
                    d["scenario_id"] = d["scenario_id"].replace(old, new)
                shutil.rmtree(tmp)
                datasets.append({"dataset": name, "seed": seed, "rows": len(joined), "columns": len(cols),
                                 "regime": REGIME[name], "base_view": {"path": str(bp.relative_to(stage)),
                                                                       "sha256": digest(bp)},
                                 "provenance": prov, "scenarios": designs})
                print(f"{name} s{seed}: {len(joined)} rows x {len(cols)} columns, {len(designs)} scenarios, "
                      f"dangling {prov['dangling']}", flush=True)
        manifest = {"builder": "sf-matcher-study/data/build_final_controlled.py", "builder_sha256": digest(Path(__file__)),
                    "fabricate": "harness/smx/controlled.py (unchanged)", "raw_root": str(RAW), "seeds": list(SEEDS),
                    "sample_size": ROWS, "datasets": datasets,
                    "protocol": "a-priori controlled design; no matcher outcomes consulted (Amendment 12)"}
        json_write(stage / "controlled-benchmark.json", manifest)
        stage.rename(OUT)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
