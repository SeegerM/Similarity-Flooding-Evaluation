"""Native Pyro for PRISMA on Windows (request): a port of code/prisma/src/main/resources/pyro/pyro_server.py,
which is Linux/Docker-only (`:` classpath, `rm -rf ./results`, one shared ./results folder).
Same interface (GET 127.0.0.1:6001/match?table=..&max_error=..; 400 unless both are given), the same Metanome CLI command
    java -cp <metanome-cli>;<pyro-distro> de.metanome.cli.App --algorithm de.hpi.isg.pyro.algorithms.Pyro --input-key inputFile
         --files <table> --output file --separator , --header --algorithm-config maxUccError:<max_error>
and the same parsing of results/*_fds* (determinants empty or wider than 3 columns skipped; lines "a|b|c" then the dependant).
Differences from the shipped service, all recorded in the run log:
  - metanome-cli 1.2-SNAPSHOT (~/.m2) instead of the shipped setup's 1.1.0 (not on this machine; no download);
    pyro-distro-1.0-SNAPSHOT-distro.jar from code/schematch/src/lib (the version the shipped setup names);
  - the Metacrate client jars (de/hpi/isg/mdms-*/0.0.3-SNAPSHOT, ~/.m2) on the classpath, which 1.2-SNAPSHOT needs;
  - jackson-annotations / jackson-core 2.12.7 (~/.m2) first on the classpath: the 1.2-SNAPSHOT jar bundles databind 2.12.7.1
    with 2.8.4 annotations/core and cannot write its results otherwise;
  - JDK 17 (`java` in the shipped container is unspecified); a 1-hour timeout per Pyro call (exit -9 = timed out);
  - every request runs in its own working folder (no shared ./results);
  - loopback only; every request is logged as one JSON line (table, max_error, exit code, #FDs returned, seconds).

    python servers/pyro_service.py --log <file> --work <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
CLI = Path.home() / ".m2" / "repository" / "de" / "metanome" / "metanome-cli" / "1.2-SNAPSHOT" / "metanome-cli-1.2-SNAPSHOT.jar"
PYRO = ROOT / "code" / "schematch" / "src" / "lib" / "pyro-distro-1.0-SNAPSHOT-distro.jar"
JAVA = Path("C:/Program Files/Java/jdk-17/bin/java.exe")
# metanome-cli 1.2-SNAPSHOT references the Metacrate client (de.hpi.isg.mdms); its 0.0.3-SNAPSHOT jars are in ~/.m2
# the 1.2-SNAPSHOT fat jar bundles jackson-databind 2.12.7.1 with jackson-annotations/-core 2.8.4 (NoClassDefFoundError
# JsonMerge when writing results); the matching 2.12.7 jars from ~/.m2 go FIRST on the classpath
JACKSON = [Path.home() / ".m2" / "repository" / "com" / "fasterxml" / "jackson" / "core" / a / "2.12.7" / f"{a}-2.12.7.jar"
           for a in ("jackson-annotations", "jackson-core")]
TIMEOUT = 3600   # a Pyro JVM that hangs (e.g. its thread pool after an exception) counts as a failure
MDMS = sorted((Path.home() / ".m2" / "repository" / "de" / "hpi" / "isg").glob("mdms-*/0.0.3-SNAPSHOT/mdms-*-0.0.3-SNAPSHOT.jar"))


def run_pyro(table: str, max_error: str, work: Path) -> tuple[list[str], int]:
    work.mkdir(parents=True)
    cmd = [str(JAVA), "-cp", ";".join(map(str, [*JACKSON, CLI, PYRO, *MDMS])), "de.metanome.cli.App", "--algorithm", "de.hpi.isg.pyro.algorithms.Pyro",
           "--input-key", "inputFile", "--files", table, "--output", "file", "--separator", ",", "--header",
           "--algorithm-config", f"maxUccError:{max_error}"]
    with (work / "pyro.log").open("w", encoding="utf-8") as log:
        try:
            code = subprocess.run(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  timeout=TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            code = -9
    res = work / "results"
    fd_files = [f for f in res.iterdir() if "_fds" in f.name] if res.is_dir() else []
    if not fd_files:
        return [], code
    parsed = []
    with fd_files[0].open(encoding="utf-8") as fh:
        for line in fh:
            fd = json.loads(line)
            dets = fd["determinant"]["columnIdentifiers"]
            if len(dets) == 0 or len(dets) > 3:
                continue
            parsed.append("|".join(d["columnIdentifier"] for d in dets))
            parsed.append(fd["dependant"]["columnIdentifier"])
    return parsed, code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--port", type=int, default=6001)
    a = ap.parse_args()
    assert CLI.is_file() and PYRO.is_file() and JAVA.is_file() and all(j.is_file() for j in JACKSON) and MDMS
    log, counter, lock = open(a.log, "a", encoding="utf-8"), itertools.count(1), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if not (u.path == "/match" and "table" in q and "max_error" in q):
                self.send_response(400)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            t0 = time.time()
            fds, code = run_pyro(q["table"], q["max_error"], Path(a.work) / f"{next(counter):05d}")
            body = "\n".join(fds).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            with lock:
                log.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "table": q["table"], "max_error": q["max_error"],
                                      "exit": code, "fds": len(fds) // 2, "seconds": round(time.time() - t0, 1)}) + "\n")
                log.flush()

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
