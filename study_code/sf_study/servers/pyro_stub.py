"""Stand-in for PRISMA's Pyro service (code/prisma/src/main/resources/pyro/pyro_server.py), option 1:
PRISMA's Database loader re-profiles every table whose shipped FD_results.txt is empty by GET :6001/match?table=..&max_error=..
(MetanomeImpl.executePyro). This answers every such request with an EMPTY body, which PRISMA parses as "no FDs", so each table
keeps exactly its shipped profile. Same interface as the original: 400 unless both `table` and `max_error` are given.
Loopback only; every answered request is logged as one JSON line (the record of skipped re-profilings). Standard library only.

    python servers/pyro_stub.py --log <file>     # serves 127.0.0.1:6001 until terminated
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--port", type=int, default=6001)
    a = ap.parse_args()
    log = open(a.log, "a", encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            ok = u.path == "/match" and "table" in q and "max_error" in q
            self.send_response(200 if ok else 400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            log.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "status": 200 if ok else 400,
                                  "table": q.get("table"), "max_error": q.get("max_error")}) + "\n")
            log.flush()

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
