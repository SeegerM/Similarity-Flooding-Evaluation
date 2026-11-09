"""Values of the registered replication of the dependency-role term (Amendment 11), read from its reports.

    results/sf_study/tuning/results/rolesreg_t2d_scores_fixedparse/report.md   T2D (corrected FD parse, Erratum 1)
    results/sf_study/tuning/results/rolesreg_t2d_scores/report.md              T2D as registered (frozen parse)
    results/sf_study/tuning/results/rolesreg_ctl_scores/report.md              Lahman/MusicBrainz (ctl), both regimes

Writes rr_numbers.tex (names rr<Corpus><Hypothesis><Est|Lo|Hi|Pos|N|Verdict>) to the value folder.
"""
import re

from paths import OUT, STUDY

RES = STUDY / "tuning" / "results"
ROW = re.compile(r"^\| (HR1|HR2|HR3|HR4a|HR4b)\b[^|]*\| [^|]* \| ([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\] \|"
                 r"(?: [^|]* \|)*? (\d+) of (\d+) \| (\w[\w ]*) \|")
HN = {"HR1": "HOne", "HR2": "HTwo", "HR3": "HThree", "HR4a": "HFourA", "HR4b": "HFourB"}


def f3(x: str) -> str:
    return "%+.3f" % float(x)


def parse(text: str, prefix: str, out: dict) -> None:
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if m:
            h, e, lo, hi, pos, n, v = m.groups()
            b = prefix + HN[h]
            out[b + "Est"], out[b + "Lo"], out[b + "Hi"] = f3(e), f3(lo), f3(hi)
            out[b + "Pos"], out[b + "N"], out[b + "Verdict"] = pos, n, v.strip().lower()


vals: dict[str, str] = {}
parse((RES / "rolesreg_t2d_scores_fixedparse" / "report.md").read_text(encoding="utf-8"), "rrTTwoD", vals)
parse((RES / "rolesreg_t2d_scores" / "report.md").read_text(encoding="utf-8"), "rrRegTTwoD", vals)
ctl = (RES / "rolesreg_ctl_scores" / "report.md").read_text(encoding="utf-8")
for reg, pre in (("opaque", "rrCtlMd"), ("plain", "rrCtlPl")):
    part = ctl.split(f"## ctl {reg}")[1].split("## ")[0]
    parse(part, pre, vals)
(OUT / "rr_numbers.tex").write_text(
    "% rr values (registered role replication, Amendment 11), parsed from the reports\n"
    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items()), encoding="utf-8")
