"""More FD-dependent reruns with the fixed HyFD (2026-10-31), cloned with clone_stage.py:

- the parallel-edge repair (fd_split = layer 1's fd + smx.sf.parallelEdges=SPLIT) on the 382 audited Valentine
  scenarios (p382fix, m382fix), on the 168 new ones (from layer1_pnew/mnew fd), and on six external corpora;
- every FD-dependent HDXSM stage (X81 fd/ucc1_fd_topk, wide fd, X42 fd_t*, parallel fd_split) on hdxsm_fdfix: the
  same 193 pairs as hdxsm_fd (the 6 pairs excluded by the FD cap stay excluded), with the stage-1 FD profiles.

    python servers/make_fix_more.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = ("fd_t1", "fd_t2", "fd_t3", "fd_t5", "fd_t10")
JOBS = [  # (base, base arm, new stage, arm, extra clone_stage arguments)
    ("parallel_plain", "fd_split", "parallelfix_p382fix", "fd_split", ["--keep-arm", "--corpus", "plain=p382fix"]),
    ("parallel_md5", "fd_split", "parallelfix_m382fix", "fd_split", ["--keep-arm", "--corpus", "md5=m382fix"]),
    ("layer1_pnew", "fd", "parallel_pnew", "fd_split", ["--props", "smx.sf.parallelEdges=SPLIT"]),
    ("layer1_mnew", "fd", "parallel_mnew", "fd_split", ["--props", "smx.sf.parallelEdges=SPLIT"]),
]
for c in ("gdcsm", "goby", "madi", "mimic", "final_controlled", "final_indep"):
    JOBS.append((f"parallel_{c}", "fd_split", f"parallelfix_{c}", "fd_split", ["--keep-arm", "--corpus", f"{c}={c}fix"]))
H = ["--keep-arm", "--corpus", "hdxsm_fd=hdxsm_fdfix"]
JOBS += [("x81_hdxsm", "fd", "x81fix_hdxsm", "fd", H), ("x81_hdxsm", "ucc1_fd_topk", "x81fix_hdxsm", "ucc1_fd_topk", H),
         ("wide_hdxsm", "fd", "widefix_hdxsm", "fd", H)]
JOBS += [("iter_x81_hdxsm", t, "iterx81fix_hdxsm", t, H) for t in T]
JOBS.append(("parallel_hdxsm", "fd_split", "parallelfix_hdxsm", "fd_split", H))

seen = set()
for base, base_arm, new, arm, extra in JOBS:
    cmd = [sys.executable, str(HERE / "clone_stage.py"), "--base", base, "--base-arm", base_arm, "--suffix", new,
           "--arm", arm, *extra] + (["--append"] if new in seen else [])
    seen.add(new)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout or r.stderr).strip())
    if r.returncode:
        sys.exit(1)
print("stages:", ",".join(dict.fromkeys(j[2] for j in JOBS)))
