"""Confirmation cohort C-CONF for the grain alignment study (GRAIN_ALIGNMENT.md s.2).

Registered in 9ea58be before this generator existed. It builds TPC-H and
AdventureWorks scenarios: normalized (N) and two denormalized variants
(D1, D2), paired N-D1, N-D2, D1-D2 (DN) and N-N (SN), in plain and recoded
regimes.

Ground truth and true grain follow from provenance, by construction. The raw
data are read only; nothing is overwritten.

    python grain_confirm.py            # writes generated/grain-confirm and grain_annotations/c_conf.json
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from smx.controlled import _encoded  # noqa: E402

RAW = Path(r"$DATA_ROOT")
OUT = HERE / "generated" / "grain-confirm"
ANNOTATION = HERE / "grain_annotations" / "c_conf.json"
SEED = 20260922
BASE_ORDERS = 1500
DROP = {"rowguid", "ModifiedDate", "SpatialLocation", "AdditionalContactInfo", "Demographics"}

# ---------------------------------------------------------------- database specs
# tables: name -> (file, key column). fks: (table, column) -> referenced table.
TPCH = {
    "name": "TPCH", "root": RAW / "TPCH", "delimiter": ";", "base": "orders",
    "tables": {"region": ("region.csv", "R_REGIONKEY"), "nation": ("nation.csv", "N_NATIONKEY"),
               "customer": ("customer.csv", "C_CUSTKEY"), "supplier": ("supplier.csv", "S_SUPPKEY"),
               "part": ("part.csv", "P_PARTKEY"), "orders": ("orders.csv", "O_ORDERKEY"),
               "lineitem": ("lineitem.csv", None)},
    "fks": {("nation", "N_REGIONKEY"): "region", ("customer", "C_NATIONKEY"): "nation",
            ("supplier", "S_NATIONKEY"): "nation", ("orders", "O_CUSTKEY"): "customer",
            ("lineitem", "L_ORDERKEY"): "orders", ("lineitem", "L_PARTKEY"): "part",
            ("lineitem", "L_SUPPKEY"): "supplier"},
    # output table -> (root table, [(join column as it appears in the wide row, role or None)])
    "variants": {
        "N": {t: (t, []) for t in ("region", "nation", "customer", "supplier", "part", "orders", "lineitem")},
        "D1": {"customer": ("customer", [("C_NATIONKEY", None), ("N_REGIONKEY", None)]),
               "supplier": ("supplier", [("S_NATIONKEY", None), ("N_REGIONKEY", None)]),
               "lineitem": ("lineitem", [("L_PARTKEY", None)]),
               "orders": ("orders", [])},
        "D2": {"orders": ("orders", [("O_CUSTKEY", None)]),
               "lineitem": ("lineitem", [("L_SUPPKEY", None)]),
               "nation": ("nation", [("N_REGIONKEY", None)]),
               "part": ("part", [])},
    },
}
AW = {
    "name": "AdventureWorks", "root": RAW / "AdventureWorks", "delimiter": ";", "base": "SalesOrderHeader",
    "tables": {"SalesOrderHeader": ("Sales.SalesOrderHeader.csv", "SalesOrderID"),
               "SalesOrderDetail": ("Sales.SalesOrderDetail.csv", "SalesOrderDetailID"),
               "Customer": ("Sales.Customer.csv", "CustomerID"),
               "Product": ("Production.Product.csv", "ProductID"),
               "ProductSubcategory": ("Production.ProductSubcategory.csv", "ProductSubcategoryID"),
               "ProductCategory": ("Production.ProductCategory.csv", "ProductCategoryID"),
               "Address": ("Person.Address.csv", "AddressID"),
               "StateProvince": ("Person.StateProvince.csv", "StateProvinceID"),
               "CountryRegion": ("Person.CountryRegion.csv", "CountryRegionCode"),
               "SalesTerritory": ("Sales.SalesTerritory.csv", "TerritoryID")},
    "fks": {("SalesOrderDetail", "SalesOrderID"): "SalesOrderHeader",
            ("SalesOrderDetail", "ProductID"): "Product",
            ("SalesOrderHeader", "CustomerID"): "Customer",
            ("SalesOrderHeader", "TerritoryID"): "SalesTerritory",
            ("SalesOrderHeader", "BillToAddressID"): "Address",
            ("SalesOrderHeader", "ShipToAddressID"): "Address",
            ("Customer", "TerritoryID"): "SalesTerritory",
            ("Product", "ProductSubcategoryID"): "ProductSubcategory",
            ("ProductSubcategory", "ProductCategoryID"): "ProductCategory",
            ("Address", "StateProvinceID"): "StateProvince",
            ("StateProvince", "CountryRegionCode"): "CountryRegion",
            ("StateProvince", "TerritoryID"): "SalesTerritory",
            ("SalesTerritory", "CountryRegionCode"): "CountryRegion"},
    "variants": {
        "N": {t: (t, []) for t in ("SalesOrderHeader", "SalesOrderDetail", "Customer", "Product",
                                   "ProductSubcategory", "ProductCategory", "Address", "StateProvince",
                                   "CountryRegion", "SalesTerritory")},
        "D1": {"sales_line": ("SalesOrderDetail", [("ProductID", None), ("ProductSubcategoryID", None),
                                                   ("ProductCategoryID", None)]),
               "orders": ("SalesOrderHeader", [("CustomerID", None)]),
               "address": ("Address", [("StateProvinceID", None), ("CountryRegionCode", None)]),
               "territory": ("SalesTerritory", [])},
        "D2": {"orders": ("SalesOrderHeader", [("BillToAddressID", "bill"), ("ShipToAddressID", "ship")]),
               "product": ("Product", [("ProductSubcategoryID", None)]),
               "customer": ("Customer", [("TerritoryID", None)]),
               "state": ("StateProvince", [("CountryRegionCode", None)]),
               "sales_detail": ("SalesOrderDetail", []),
               "category": ("ProductCategory", [])},
    },
}
PAIRINGS = [("N", "D1", "DN"), ("N", "D2", "DN"), ("D1", "D2", "DN"), ("N", "N", "SN")]
REGIMES = ("plain", "recoded")


def rank(key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}\0{key}".encode()).digest()[:8], "big")


def read_rows(db, table, keep=None):
    path = db["root"] / db["tables"][table][0]
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter=db["delimiter"])
        header = [h.strip() for h in next(reader)]
        cols = [i for i, h in enumerate(header) if h not in DROP]
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{path}: ragged row")
            rec = {header[i]: row[i].strip() for i in cols}
            if keep is None or keep(rec):
                yield rec


def sample(db):
    """Base-order sample with FK closure."""
    base = db["base"]
    base_key = db["tables"][base][1]
    rows = {base: heapq.nsmallest(BASE_ORDERS, read_rows(db, base), key=lambda r: rank(r[base_key]))}
    keys = {base: {r[base_key] for r in rows[base]}}
    # children referencing the base (detail rows)
    for (table, col), ref in db["fks"].items():
        if ref == base and table not in rows:
            rows[table] = list(read_rows(db, table, lambda r, c=col: r[c] in keys[base]))
    changed = True
    while changed:                            # parents, until closure
        changed = False
        for (table, col), ref in db["fks"].items():
            if table not in rows or ref == base:
                continue
            needed = {r[col] for r in rows[table] if r[col]}
            have = keys.get(ref, set())
            if needed - have:
                ref_key = db["tables"][ref][1]
                want = needed | have
                rows[ref] = list(read_rows(db, ref, lambda r, k=ref_key: r[k] in want))
                keys[ref] = {r[ref_key] for r in rows[ref]}
                changed = True
    for table in db["tables"]:
        rows.setdefault(table, [])
    return rows


def build_variant(db, rows, variant):
    """Output tables as (header provenance, rows). Provenance: list of (base, col, role) per column."""
    index = {t: {r[k]: r for r in rows[t]} for t, (_, k) in db["tables"].items() if k}
    out = {}
    for name, (root, joins) in db["variants"][variant].items():
        prov = [[(root, c, None)] for c in rows[root][0].keys()] if rows[root] else []
        where = {c: i for i, c in enumerate(rows[root][0].keys())}
        table_rows = [list(r.values()) for r in rows[root]]
        owner = {c: root for c in rows[root][0].keys()}
        for join_col, role in joins:
            src_table = owner[join_col]
            ref = db["fks"][(src_table, join_col)]
            ref_key = db["tables"][ref][1]
            ref_cols = [c for c in (rows[ref][0].keys() if rows[ref] else []) if c != ref_key]
            prov[where[join_col]].append((ref, ref_key, role))
            j = where[join_col]
            for c in ref_cols:
                label = f"{role}_{c}" if role else c
                if label in where:
                    label = f"{ref}_{c}"
                where[label] = len(prov)
                prov.append([(ref, c, role)])
                owner[label] = ref
            for r in table_rows:
                hit = index[ref].get(r[j], {})
                r.extend(hit.get(c, "") for c in ref_cols)
        out[name] = (prov, table_rows, list(where))
    return out


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def target_label(db_name, raw_label):
    if db_name == "TPCH":
        parts = raw_label.split("_", 1)
        return (parts[1] if len(parts) == 2 and len(parts[0]) == 1 else raw_label).lower()
    return snake(raw_label)


def labels_for(db, table_prov, raw_labels, side, regime, pairing, table):
    if regime == "recoded":
        return [f"{side[0]}_{hashlib.sha256(f'{SEED}|{db['name']}|{pairing}|{side}|{table}|{i}'.encode()).hexdigest()[:16]}"
                for i in range(len(raw_labels))]
    names = []
    for raw, prov in zip(raw_labels, table_prov):
        base, col, role = prov[0]
        name = raw if side == "source" else target_label(db["name"], raw)
        names.append(name)
    seen = {}
    for i, n in enumerate(names):
        seen.setdefault(n, []).append(i)
    for n, idx in seen.items():
        if len(idx) > 1:
            for i in idx:
                base, col, role = table_prov[i][0]
                prefix = role or (base if side == "source" else snake(base))
                names[i] = f"{prefix}_{n}"
    if len(set(names)) != len(names):
        raise ValueError(f"{db['name']}/{table}: labels still collide")
    return names


def corresponds(sp, tp) -> bool:
    """Shared (base table, base column); roles must agree only when both sides carry one."""
    return any(b1 == b2 and c1 == c2 and (r1 is None or r2 is None or r1 == r2)
               for b1, c1, r1 in sp for b2, c2, r2 in tp)


def entity(db, prov):
    """True entity: base table, or the table a key/FK column identifies (by construction)."""
    base, col, role = prov[0]
    for b, c, _ in prov:
        if (b, c) in db["fks"]:
            return db["fks"][(b, c)]
    return base


def write_table(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    annotations = {}
    for db in (TPCH, AW):
        rows = sample(db)
        counts = {t: len(v) for t, v in rows.items()}
        print(db["name"], counts, flush=True)
        variants = {v: build_variant(db, rows, v) for v in db["variants"]}
        for s_var, t_var, norm in PAIRINGS:
            pairing = f"{s_var}_{t_var}"
            for regime in REGIMES:
                scen = f"{db['name']}_{pairing}_{regime}"
                folder = OUT / scen
                sides = {"source": variants[s_var], "target": variants[t_var]}
                labelled = {}
                ann = {"path": str(folder.relative_to(HERE.parent)), "regime": "ORIG" if regime == "plain" else "ENC",
                       "normalization": norm, "dataset": db["name"], "source": {}, "target": {}}
                for side, tables in sides.items():
                    labelled[side] = {}
                    for table, (prov, trows, raw) in tables.items():
                        names = labels_for(db, prov, raw, side, regime, pairing, table)
                        if regime == "recoded":
                            enc = [[_encoded(v, side, f"{db['name']}|{pairing}|{table}|{raw[i]}") for i, v in enumerate(r)]
                                   for r in trows]
                        else:
                            enc = trows
                        write_table(folder / side / f"{table}.csv", names, enc)
                        labelled[side][table] = list(zip(names, prov))
                        ann[side][table] = {n: {"entity": entity(db, p), "role": [table, entity(db, p)
                                                if not p[0][2] else f"{p[0][2]}:{entity(db, p)}"],
                                                "qualified": "+".join(f"{b}.{c}" for b, c, _ in p)}
                                            for n, p in zip(names, prov)}
                # ground truth: shared (base, column) provenance, role-aware for role-played copies
                gt_pairs = 0
                for s_table, s_cols in labelled["source"].items():
                    for t_table, t_cols in labelled["target"].items():
                        grid = [[int(corresponds(sp, tp)) for _, tp in t_cols] for _, sp in s_cols]
                        if any(any(row) for row in grid):
                            gt_pairs += sum(map(sum, grid))
                            _write_grid(folder / "ground_truth" / f"{s_table}___{t_table}.csv", grid)
                ann["checks"] = {"gt_pairs": gt_pairs, "rows": counts}
                annotations[scen] = ann
                print(f"  {scen}: {gt_pairs} GT pairs", flush=True)
    ANNOTATION.write_text(json.dumps({"cohort": "C-CONF", "scenarios": annotations}, indent=1,
                                     ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {ANNOTATION.name}")
    return 0


def _write_grid(path: Path, grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(grid)


if __name__ == "__main__":
    raise SystemExit(main())
