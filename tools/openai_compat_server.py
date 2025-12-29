#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

def _read_json(req):
    n = int(req.headers.get("Content-Length", "0") or "0")
    if n <= 0:
        return None
    raw = req.rfile.read(n)
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None

def _send_json(req, code, obj):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req.send_response(code)
    req.send_header("Content-Type", "application/json")
    req.send_header("Content-Length", str(len(body)))
    req.end_headers()
    req.wfile.write(body)

def _run_hf(python_exe, script, model_id, model_path, device, max_new, prompt, temperature):
    cmd = [
        python_exe,
        script,
        "--device", device,
        "--max-new", str(max_new),
        "--prompt", prompt,
    ]

    if model_path:
        cmd += ["--model", model_path]
    else:
        cmd += ["--model", model_id]

    if temperature is not None:
        cmd += ["--temperature", str(temperature)]

    p = subprocess.run(cmd, capture_output=True, text=True)
    stdout = (p.stdout or "").strip()
    stderr = (p.stderr or "").strip()

    details = {
        "returncode": p.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "model": model_path or model_id,
        "device": device,
        "max_new": max_new,
    }

    if p.returncode != 0:
        return None, details

    if not stdout:
        return None, details

    try:
        out = json.loads(stdout)
    except Exception:
        details["parse_error"] = "stdout was not valid JSON"
        return None, details

    return out, details

class Handler(BaseHTTPRequestHandler):
    server_version = "ClocherOpenAICompat/1.0"

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/v1/models":
            data = [{"id": self.server.model_id, "object": "model", "owned_by": "local"}]
            return _send_json(self, 200, {"object": "list", "data": data})
        return _send_json(self, 404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/v1/chat/completions":
            return _send_json(self, 404, {"error": {"message": "not found", "type": "not_found"}})

        payload = _read_json(self)
        if not payload:
            return _send_json(self, 400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}})

        msgs = payload.get("messages") or []
        prompt = ""
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                prompt = m.get("content") or ""
                break

        max_new = payload.get("max_tokens", 64)
        try:
            max_new = int(max_new)
        except Exception:
            max_new = 64
        if max_new < 0:
            max_new = 0

        temperature = payload.get("temperature", None)
        try:
            if temperature is not None:
                temperature = float(temperature)
        except Exception:
            temperature = None

        out, details = _run_hf(
            python_exe=self.server.python_exe,
            script=self.server.script_path,
            model_id=self.server.model_id,
            model_path=self.server.model_path,
            device=self.server.device,
            max_new=max_new,
            prompt=prompt,
            temperature=temperature,
        )

        if not out:
            return _send_json(self, 500, {
                "error": {
                    "message": "generation failed",
                    "type": "server_error",
                    "details": details,
                }
            })

        if isinstance(out, dict) and out.get("error"):
            return _send_json(self, 500, {
                "error": {
                    "message": "generation failed",
                    "type": "server_error",
                    "details": out,
                }
            })

        text = ""
        if isinstance(out, dict):
            text = out.get("text") or ""

        if not text:
            return _send_json(self, 500, {
                "error": {
                    "message": "generation returned empty text",
                    "type": "server_error",
                    "details": out,
                }
            })

        resp = {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.server.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }
        return _send_json(self, 200, resp)

    def log_message(self, fmt, *args):
        return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--python", dest="python_exe", default="python3")
    ap.add_argument("--script", required=True)
    ap.add_argument("--model-id", default="gpt-oss-20b")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--model", default="", help="Alias for --model-path")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_path = args.model_path or args.model
    httpd = HTTPServer((args.host, args.port), Handler)
    httpd.python_exe = args.python_exe
    httpd.script_path = args.script
    httpd.model_id = args.model_id
    httpd.model_path = model_path
    httpd.device = args.device

    httpd.serve_forever()

if __name__ == "__main__":
    main()
