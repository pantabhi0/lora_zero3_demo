"""Phase 6 — simple web GUI reading Phase 5 CSVs (TUI-equivalent features).

One local server per machine (A=rank0, B=rank1). Browser polls --refresh
seconds; the page shows the same per-step data the TUI showed: loss,
samples_processed, compute_ms, comm_ms, live during or after a run. All rows
are shown (the TUI's 40-row cap was a terminal-height limit).

Usage:
    uv run python -m src.webui --csv logs/run_default_rank0.csv --port 8000
    uv run python -m src.webui --csv logs/run_default_rank1.csv --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.tui import read_rows, timing_path_for

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HeteroTrain</title>
<style>
body{font-family:ui-monospace,Menlo,monospace;background:#111;color:#ddd;margin:2rem}
h1{font-size:1.1rem;color:#7fd}
#meta{color:#888;margin-bottom:.5rem}
table{border-collapse:collapse;width:100%}
th,td{padding:.3rem .6rem;border-bottom:1px solid #333;text-align:right}
th{color:#9df;position:sticky;top:0;background:#111}
td:nth-child(2){color:#fa8}
td:nth-child(4),td:nth-child(5){color:#8d8}
</style>
</head>
<body>
<h1 id="title">loading…</h1>
<div id="meta"></div>
<table><thead><tr>
<th>step</th><th>loss</th><th>samples</th><th>compute_ms</th><th>comm_ms</th>
</tr></thead><tbody id="rows"></tbody></table>
<script>
const REFRESH_MS = __REFRESH_MS__;
async function tick(){
  try{
    const r = await fetch('/api/rows');
    const d = await r.json();
    document.getElementById('title').textContent = d.title;
    document.getElementById('meta').textContent =
      d.rows.length + ' rows · ' + new Date().toLocaleTimeString();
    const tb = document.getElementById('rows');
    tb.innerHTML = d.rows.map(x =>
      '<tr><td>' + x.step + '</td><td>' + x.loss + '</td><td>' + x.samples +
      '</td><td>' + x.compute_ms + '</td><td>' + x.comm_ms + '</td></tr>'
    ).join('');
  }catch(e){
    document.getElementById('meta').textContent = 'waiting for server…';
  }
}
setInterval(tick, REFRESH_MS);
tick();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    run_csv: Path = Path()
    refresh_ms: int = 1000

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            page = PAGE.replace("__REFRESH_MS__", str(self.refresh_ms)).encode()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if self.path == "/api/rows":
            # Timing CSV steps are 0-based, run CSV steps are 1-based:
            # merge by POSITION, not by step key.
            timing = read_rows(timing_path_for(self.run_csv))
            rows = []
            for i, r in enumerate(read_rows(self.run_csv)):
                t = timing[i] if i < len(timing) else {}
                rows.append({
                    "step": r.get("step", ""),
                    "loss": r.get("loss", ""),
                    "samples": r.get("samples_processed", ""),
                    "compute_ms": t.get("compute_ms", ""),
                    "comm_ms": t.get("comm_ms", ""),
                })
            payload = json.dumps({
                "title": f"rank CSV — {self.run_csv.name}",
                "rows": rows,
            }).encode()
            self._send(200, payload, "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="run CSV, e.g. logs/run_default_rank0.csv")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--refresh", type=float, default=1.0, help="browser poll interval (s)")
    args = p.parse_args()

    run_csv = Path(args.csv)
    if not run_csv.exists():
        raise SystemExit(f"missing CSV: {run_csv}")

    Handler.run_csv = run_csv
    Handler.refresh_ms = max(250, int(args.refresh * 1000))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"HeteroTrain web GUI on http://{args.host}:{args.port}  (CSV: {run_csv})",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()