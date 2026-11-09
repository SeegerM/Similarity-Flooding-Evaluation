"""Scenario -> cluster maps of the external corpora (shared tables, studies), cached as results/derived/clusters.json
for the experiments' cluster bootstrap. Needs the study's research tree with the raw benchmark manifests
(<RESEARCH_ROOT> below); the artifact itself reads only the cached JSON."""
import csv
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else '<RESEARCH_ROOT>')
ST = R / 'sf-matcher-study'
for p in (R / 'harness', ST / 'tuning', ST, ST / 'reference', ST / 'layer3b'):
    sys.path.insert(0, str(p))
from explore_x81 import clusters  # noqa: E402

rows = list(csv.DictReader(open(Path(__file__).resolve().parents[2] / 'results' / 'derived' / 'recall' /
                                'w4_keyterm_ext_rows.csv', encoding='utf-8')))
out = {}
for rep in ('madi_indep', 'madi_products', 'goby', 'hdxsm', 'gdcsm'):
    scen = sorted({r['scenario'] for r in rows if r['corpus'] == rep})
    out[rep] = clusters(rep, scen)
    print(rep, len(scen), 'scenarios', len(set(out[rep].values())), 'clusters')
dst = Path(__file__).resolve().parents[2] / 'results' / 'derived' / 'clusters.json'
dst.write_text(json.dumps(out, indent=1, sort_keys=True), encoding='utf-8')
