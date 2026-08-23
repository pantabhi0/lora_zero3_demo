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
import mimetypes
import os
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.tui import read_rows, timing_path_for
from src.web.inference import InferenceService

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HeteroTrain</title>
 <link rel="stylesheet" href="/static/style.css">
</head>
<body>
<nav><button id="metricsTab" class="active" onclick="showTab('metrics')">Metrics</button><button id="inferTab" onclick="showTab('infer')">Inference</button></nav>
<div class="panel"><label>Analysis scale <select id="scale" onchange="loadAnalysis()"><option>1k</option><option>long</option></select></label> <a href="/download-all">Download all local artifacts</a></div>
<section id="metrics">
<h1 id="title">loading…</h1>
<div id="meta"></div>
<div id="pairMeta" class="muted"></div>
<div id="academicSummary" class="panel"><h2>Academic Summary</h2><div id="summaryTable" class="muted">loading…</div></div>
<table><thead><tr>
<th>step</th><th>loss</th><th>samples</th><th>compute_ms</th><th>comm_ms</th>
</tr></thead><tbody id="rows"></tbody></table>
<div id="figures" class="grid"></div>
</section>
<section id="infer" class="hidden">
<div class="panel"><label>Adapter <select id="adapter"></select></label><label>Max tokens <input id="maxTokens" type="number" value="512"></label><label>Temperature <input id="temperature" type="number" step="0.1" value="0.7"></label><label>Top-p <input id="topP" type="number" step="0.05" value="0.9"></label><label>Repetition <input id="repetition" type="number" step="0.1" value="1.2"></label><label>Seed <input id="seed" type="number" value="42"></label></div>
<div class="panel"><textarea id="prompt" rows="4" style="width:100%" placeholder="Describe a coding task..."></textarea><button id="send" onclick="sendPrompt()">Send</button><span id="status" class="muted"></span></div>
<div class="grid"><div class="panel"><h2>BASE</h2><div id="base" class="answer"></div></div><div class="panel"><h2>FINETUNED</h2><div id="finetuned" class="answer"></div></div></div>
</section>
<script>
const REFRESH_MS = __REFRESH_MS__;
function showTab(tab){document.getElementById('metrics').classList.toggle('hidden',tab!=='metrics');document.getElementById('infer').classList.toggle('hidden',tab!=='infer');document.getElementById('metricsTab').classList.toggle('active',tab==='metrics');document.getElementById('inferTab').classList.toggle('active',tab==='infer');}
function safeText(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function renderMarkdown(s){let x=safeText(s);x=x.replace(/```[a-zA-Z0-9_-]*\n?([\s\S]*?)```/g,'<pre><code>$1</code></pre>');x=x.replace(/`([^`]+)`/g,'<code>$1</code>');x=x.replace(/^### (.*)$/gm,'<h3>$1</h3>');x=x.replace(/^## (.*)$/gm,'<h2>$1</h2>');x=x.replace(/^# (.*)$/gm,'<h1>$1</h1>');x=x.replace(/\n/g,'<br>');return x;}
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
async function loadAdapters(){const r=await fetch('/api/inference/adapters');if(!r.ok){document.getElementById('inferTab').disabled=true;return;}const d=await r.json();document.getElementById('adapter').innerHTML=d.tags.map(x=>'<option>'+safeText(x)+'</option>').join('');}
async function loadAnalysis(){const scale=document.getElementById('scale').value;const r=await fetch('/api/analysis?scale='+scale);if(!r.ok)return;const d=await r.json();document.getElementById('figures').innerHTML=d.figures.map(x=>'<div class="panel"><h3>'+safeText(x.name)+'</h3><details><summary>What this measures</summary><p>'+safeText(x.description)+'</p></details><img class="figure" src="'+safeText(x.url)+'"></div>').join('');const p=await fetch('/api/pair?scale='+scale);if(p.ok){const pair=await p.json();document.getElementById('pairMeta').textContent=pair.runs.map(x=>x.run_tag+': '+x.tokens_per_sec+' tok/s, val_loss='+x.final_val_loss).join(' | ');}}
async function loadAcademicSummary(){const r=await fetch('/api/academic-summary');if(!r.ok)return;const d=await r.json();let h='<table><thead><tr><th>run</th><th>world</th><th>samples</th><th>wall_s</th><th>tokens/s</th><th>val_loss</th><th>val_PPL</th></tr></thead><tbody>';h+=d.rows.map(x=>'<tr><td>'+safeText(x.run_tag)+'</td><td>'+safeText(x.world_size)+'</td><td>'+safeText(x.total_samples)+'</td><td>'+safeText(x.wall_clock_s)+'</td><td>'+safeText(x.tokens_per_sec)+'</td><td>'+safeText(x.final_val_loss)+'</td><td>'+safeText(x.final_val_perplexity)+'</td></tr>').join('');document.getElementById('summaryTable').innerHTML=h+'</tbody></table>';}
async function sendPrompt(){const button=document.getElementById('send');button.disabled=true;document.getElementById('status').textContent='generating...';try{const r=await fetch('/api/inference/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tag:document.getElementById('adapter').value,instruction:document.getElementById('prompt').value,max_new_tokens:Number(document.getElementById('maxTokens').value),temperature:Number(document.getElementById('temperature').value),top_p:Number(document.getElementById('topP').value),repetition_penalty:Number(document.getElementById('repetition').value),seed:Number(document.getElementById('seed').value)})});const d=await r.json();if(!r.ok)throw new Error(d.error);document.getElementById('base').innerHTML=renderMarkdown(d.base);document.getElementById('finetuned').innerHTML=renderMarkdown(d.finetuned);document.getElementById('status').textContent='ready';}catch(e){document.getElementById('status').textContent=e.message;}finally{button.disabled=false;}}
setInterval(tick, REFRESH_MS);
tick();
loadAdapters();
loadAnalysis();
loadAcademicSummary();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    run_csv: Path = Path()
    refresh_ms: int = 1000
    inference: InferenceService | None = None
    analysis_dir: Path | None = None
    pair_scale: str | None = None

    FIGURE_DESCRIPTIONS = {
        "loss_tokens": "Training loss measures response-token prediction error on optimization batches. Held-out validation loss measures response-token prediction error on fixed examples never used for training. The x-axis is tokens_seen so configurations are compared at equal data exposure rather than equal optimizer step.",
        "throughput": "Tokens per second measures processed non-padding sequence tokens divided by end-to-end wall time. It measures training throughput, not model quality.",
        "speedup_efficiency": "Speedup is baseline wall time divided by 2-node wall time. Parallel efficiency is speedup divided by ideal two-worker speedup of 2. Values near 1 indicate little distributed overhead.",
        "gpu_utilization": "GPU utilization is the percentage reported by nvidia-smi at approximately 1 Hz. It indicates GPU busy time, not achieved FLOPS or model quality.",
        "breakdown": "The breakdown separates de-overlapped compute, communication, loader/H2D data preparation, validation, and residual time per optimizer update. It explains where user-visible runtime is spent.",
        "bandwidth": "Effective bandwidth is estimated from synchronized LoRA gradient payload bytes divided by measured communication duration. The 1 Gbps ceiling is 125 MB/s. This is not packet-level network measurement.",
        "peak_vram": "Peak VRAM reports maximum PyTorch allocated and reserved device memory. Allocated is live tensor memory; reserved is memory held by the CUDA caching allocator.",
        "comm_stability": "Communication stability shows per-update all-reduce duration and a rolling mean. It exposes NCCL warm-up and steady-state synchronization variability.",
        "final_quality": "Final held-out validation loss is the primary quality metric for the saved final adapter. Lower loss means better response-token prediction on unseen examples. Perplexity is exp(validation loss) when finite.",
        "speedup_consistency": "Speedup consistency compares measured 1-node/2-node wall-time ratios at the two tested scales. It is not a scaling curve and does not claim behavior for additional GPU counts.",
    }

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/static/"):
            path = Path(__file__).parent / "web" / "static" / self.path.removeprefix("/static/")
            if not path.is_file() or ".." in path.parts:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            return
        if self.path.startswith("/analysis/"):
            if self.analysis_dir is None:
                self._send(404, b"not found", "text/plain")
                return
            path = (self.analysis_dir / self.path.removeprefix("/analysis/")).resolve()
            root = self.analysis_dir.resolve()
            if not path.is_file() or root not in path.parents:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            return
        if self.path == "/download-all":
            self._download_all()
            return
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
        if self.path.startswith("/api/pair"):
            scale = parse_qs(urlparse(self.path).query).get("scale", [self.pair_scale or "1k"])[0]
            if scale not in ("1k", "long"):
                self._send(400, b"invalid scale", "application/json")
                return
            tags = {"1k": ("baseline_1k", "2node_1k"), "long": ("baseline_long", "2node_long")}[scale]
            summaries = []
            for tag in tags:
                path = Path("logs") / "academic_summary.csv"
                match = next((row for row in read_rows(path) if row.get("run_tag") == tag), {})
                summaries.append(match)
            self._send(200, json.dumps({"scale": scale, "runs": summaries}).encode(), "application/json")
            return
        if self.path == "/api/inference/adapters":
            if self.inference is None:
                self._send(503, b'{"error":"inference unavailable"}', "application/json")
                return
            self._send(200, json.dumps({"tags": self.inference.tags()}).encode(), "application/json")
            return
        if self.path.startswith("/api/analysis"):
            if self.analysis_dir is None:
                self._send(404, b"not found", "application/json")
                return
            scale = parse_qs(urlparse(self.path).query).get("scale", ["1k"])[0]
            folder = self.analysis_dir / scale
            figures = []
            if folder.is_dir():
                for path in sorted(folder.glob("*.png")):
                    prefix = next((key for key in self.FIGURE_DESCRIPTIONS
                                   if path.stem.startswith(key)), "")
                    figures.append({
                        "name": path.stem,
                        "url": f"/analysis/{scale}/{path.name}",
                        "description": self.FIGURE_DESCRIPTIONS.get(
                            prefix, "This figure is generated from measured campaign artifacts."
                        ),
                    })
            self._send(200, json.dumps({"figures": figures}).encode(), "application/json")
            return
        if self.path == "/api/academic-summary":
            self._send(200, json.dumps({
                "rows": read_rows(Path("logs") / "academic_summary.csv")
            }).encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def _download_all(self) -> None:
        archive = Path("/tmp") / "heterotrain-artifacts.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for pattern in ("*.csv", "validation_*.csv"):
                for path in Path("logs").glob(pattern):
                    bundle.write(path, path.as_posix())
            for path in Path("logs").glob("*/metadata.json"):
                bundle.write(path, path.as_posix())
            for path in Path("logs").glob("*/config.json"):
                bundle.write(path, path.as_posix())
            if self.analysis_dir and self.analysis_dir.exists():
                for path in self.analysis_dir.rglob("*"):
                    if path.is_file():
                        bundle.write(path, path.as_posix())
            for path in Path("configs").glob("resolved/*.yaml"):
                bundle.write(path, path.as_posix())
        body = archive.read_bytes()
        self._send(200, body, "application/zip")

    def do_POST(self) -> None:
        if self.path != "/api/inference/generate" or self.inference is None:
            self._send(404, b"not found", "text/plain")
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body)
            result = self.inference.generate_pair(
                payload["tag"], payload["instruction"], payload.get("max_new_tokens", 512),
                payload.get("temperature", 0.7), payload.get("top_p", 0.9),
                payload.get("repetition_penalty", 1.2), payload.get("seed", 42),
            )
            self._send(200, json.dumps(result).encode(), "application/json")
        except Exception as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None, help="run CSV, e.g. logs/run_default_rank0.csv")
    p.add_argument("--analysis-dir", default="logs/analysis")
    p.add_argument("--pair", choices=["1k", "long"], default=None)
    p.add_argument("--inference", action="store_true", help="enable A-side GPU inference page")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--refresh", type=float, default=1.0, help="browser poll interval (s)")
    args = p.parse_args()

    run_csv = Path(args.csv) if args.csv else Path("logs/run_baseline_1k_rank0.csv")
    if args.pair:
        baseline = "baseline_1k" if args.pair == "1k" else "baseline_long"
        run_csv = Path("logs") / f"run_{baseline}_rank0.csv"
    if args.pair is None and not run_csv.exists():
        raise SystemExit(f"missing CSV: {run_csv}")

    Handler.run_csv = run_csv
    Handler.refresh_ms = max(250, int(args.refresh * 1000))
    if args.inference and os.environ.get("RANK", "0") != "0":
        raise SystemExit("inference page is available on rank 0 / laptop A only")
    Handler.inference = InferenceService() if args.inference and Path("logs").exists() else None
    Handler.analysis_dir = Path(args.analysis_dir)
    Handler.pair_scale = args.pair
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"HeteroTrain web GUI on http://{args.host}:{args.port}  (CSV: {run_csv})",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
