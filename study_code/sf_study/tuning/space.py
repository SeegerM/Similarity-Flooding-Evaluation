"""The pre-registered SF tuning space (PREREGISTRATION.md): seed panel and configurations.

A configuration is (operator, coefficient policy, seed attachment, dependency strength). Matcher
parameters go into second_line_matchers.yaml; smx.sf.* switches are JVM system properties, which
is why every configuration is its own Schematch run.
"""

from __future__ import annotations

SEED_PANEL = ["oracle_noise050", "levenshtein", "prefix_suffix", "semstruct", "pmf", "ks",
              "kang_naughton", "role_gcov", "constant"]

# seeds that are FILE seeds in the seed study (their matrices are already a seed directory)
FILE_SEED_DIRS = {"semstruct": "seeds_imported/{regime}/semstruct"}

OPERATORS = ([("fixpoint", f) for f in ("BASIC", "A", "B", "C")]
             + [("restart", a) for a in (0.15, 0.30, 0.50, 0.70, 0.85)])
COEFFS = ("INV_PROD", "INV_AVG")
ATTACHMENTS = ("NAME", "ID")
FD_STRENGTH = ("full", "damped", "off")

PRODUCTION = {"operator": "fixpoint", "value": "C", "coeff": "INV_PROD", "attach": "NAME", "fd": "full"}
# Amendment 1: restart at alpha = 0 returns the *attached* seed, and SF reads its output from the
# column (ID) pairs; with NAME attachment the column pairs then receive nothing. The control and the
# alpha curve therefore attach the seed to ID pairs, as the harness's restart arms always did.
NO_PROPAGATION = {"operator": "restart", "value": 0.0, "coeff": "INV_PROD", "attach": "ID", "fd": "full"}
ALPHA_CURVE_ATTACH = "ID"

BASE_SF = {
    "graphEncoding": "QUICK",
    "indFilterThreshold": "",
    "uccFilterThreshold": "",
    "fdFilterThreshold": "0.0",
    "dependencyTopK": "",
    "fdAllowListFile": "",
    "initialMatrix": "FILE",
}


def name(cfg: dict) -> str:
    op = f"fp{cfg['value']}" if cfg["operator"] == "fixpoint" else f"r{float(cfg['value']):.2f}"
    return f"{op}__{cfg['coeff']}__{cfg['attach']}__fd{cfg['fd']}"


def configurations() -> list[dict]:
    out = []
    for operator, value in OPERATORS:
        for coeff in COEFFS:
            for attach in ATTACHMENTS:
                for fd in FD_STRENGTH:
                    out.append({"operator": operator, "value": value, "coeff": coeff,
                                "attach": attach, "fd": fd})
    out.append(dict(NO_PROPAGATION))
    return out


def sf_params(cfg: dict) -> dict:
    p = dict(BASE_SF)
    p["propCoeffPolicy"] = cfg["coeff"]
    # with restart the historical fixpoint is not used; C is recorded for completeness
    p["fixpoint"] = cfg["value"] if cfg["operator"] == "fixpoint" else "C"
    p["constraintWeighting"] = "NONE" if cfg["fd"] == "full" else "CONSTANT"
    return p


def system_properties(cfg: dict) -> dict:
    props = {"smx.sf.seedAttachment": cfg["attach"]}
    if cfg["operator"] == "restart":
        props["smx.sf.restartAlpha"] = f"{float(cfg['value']):.2f}"
    if cfg["fd"] == "damped":
        props["smx.sf.constantWeight"] = "0.35"
    elif cfg["fd"] == "off":
        props["smx.sf.constantWeight"] = "0.0"
    return props


def is_production(cfg: dict) -> bool:
    return all(cfg[k] == v for k, v in PRODUCTION.items())
