"""The FD-dependent arms of the external-corpus stages (X81, X82 wide panel, X42 iteration on X81, X54/X55 and their
iteration stages), cloned unchanged onto the hosts' stage-1 corpora <corpus>fix (FD profiles from the fixed HyFD,
harness/generated_hyfdfix; tables identical). HDXSM is left out: its runs use the FD-capped hdxsm_fd (193 pairs), and
the cap was decided on the old FD counts.

    python servers/make_fix_externals.py            # writes tuning/configs/<stage>fix_<corpus>/
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = ("fd_t1", "fd_t2", "fd_t3", "fd_t5", "fd_t10")
JOBS = []   # (base stage, new stage, corpus, arms)
for c in ("gdcsm", "goby", "madi", "mimic"):
    JOBS += [(f"x81_{c}", f"x81fix_{c}", c, ("fd", "ucc1_fd_topk")),
             (f"wide_{c}", f"widefix_{c}", c, ("fd",)),
             (f"iter_x81_{c}", f"iterx81fix_{c}", c, T)]
for c in ("final_controlled", "final_indep"):
    JOBS += [(f"wide_{c}", f"widefix_{c}", c, ("fd",))]
JOBS += [("x54_synthea", "x54fix_synthea", "synthea", ("fd",)), ("x55_indep2", "x55fix_indep2", "indep2", ("fd",)),
         ("iter_x54_synthea", "iterx54fix_synthea", "synthea", T), ("iter_x55_indep2", "iterx55fix_indep2", "indep2", T)]

for base, new, corpus, arms in JOBS:
    for i, arm in enumerate(arms):
        cmd = [sys.executable, str(HERE / "clone_stage.py"), "--base", base, "--base-arm", arm, "--suffix", new,
               "--arm", arm, "--keep-arm", "--corpus", f"{corpus}={corpus}fix"] + (["--append"] if i else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        print((r.stdout or r.stderr).strip())
        if r.returncode:
            sys.exit(1)
print("stages:", ",".join(dict.fromkeys(n for _, n, _, _ in JOBS)))
