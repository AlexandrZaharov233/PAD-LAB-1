from __future__ import annotations

import json
import threading
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scada_rag.common import DATA, ROOT, config, documents, read_json
from scada_rag.generation.ollama import request_json


class Application:
    def __init__(self):
        self.pipeline = None
        self.lock = threading.Lock()
        self.state = {"busy": False, "ready": False, "logs": [], "error": None}

    def log(self, message):
        self.state["logs"] = (self.state["logs"] + [str(message)])[-100:]
        print(message, flush=True)

    def prepare(self, refresh=True):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Дождитесь завершения текущей операции")
        self.state.update({"busy": True, "ready": False, "error": None})

        def work():
            try:
                if refresh:
                    from scada_rag.grabber.collector import collect
                    result = collect(self.log)
                    failures = [x for x in result if x["status"] == "error"]
                    if failures:
                        raise RuntimeError("Не все источники обновлены: " + json.dumps(failures, ensure_ascii=False))
                from scada_rag.pipeline import Pipeline
                self.pipeline = None
                import gc
                gc.collect()
                self.log("Загрузка embedding-модели и построение индекса...")
                self.pipeline = Pipeline(log=self.log)
                self.state["ready"] = True
                self.log("Система готова. Можно задавать вопросы.")
            except Exception as exc:
                self.state["error"] = str(exc)
                self.log("Ошибка: " + str(exc))
                traceback.print_exc()
            finally:
                self.state["busy"] = False
                self.lock.release()
        threading.Thread(target=work, daemon=True).start()

    def status(self):
        cfg = config()
        docs = documents()
        try:
            tags = request_json(cfg["ollama_url"], "/api/tags", timeout=2)
            models = [m["name"] for m in tags.get("models", [])]
            ollama = {"running": True, "model_ready": cfg["llm_model"] in models}
        except RuntimeError:
            ollama = {"running": False, "model_ready": False}
        return {**self.state, "documents": len(docs), "chunks": self.pipeline.store.meta["chunk_count"] if self.pipeline else 0,
                "config": cfg, "ollama": ollama,
                "sources": [{k: d.get(k) for k in ("id", "source", "title", "url", "updated_at", "note")} for d in docs],
                "grabber": read_json(DATA / "state.json", {})}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/status":
                return self.reply(app.status())
            if path == "/api/results":
                files = sorted((ROOT / "experiments").rglob("comparison.json"), reverse=True)
                return self.reply([{"path": str(p.relative_to(ROOT)), "rows": read_json(p)} for p in files[:15]])
            allowed = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path not in allowed:
                return self.reply({"error": "Не найдено"}, 404)
            file = Path(__file__).parent / "web" / allowed[path]
            payload = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[file.suffix] + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            try:
                host = self.headers.get("Host", "").split(":")[0]
                origin = self.headers.get("Origin")
                if host not in {"localhost", "127.0.0.1"} or (origin and urllib.parse.urlsplit(origin).hostname not in {"localhost", "127.0.0.1"}):
                    return self.reply({"error": "Разрешены запросы только с localhost"}, 403)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 20000:
                    return self.reply({"error": "Слишком большой запрос"}, 413)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/prepare":
                    app.prepare(refresh=True)
                    return self.reply({"started": True}, 202)
                if self.path == "/api/ask":
                    if not app.state["ready"]:
                        return self.reply({"error": "Сначала подготовьте базу документов"}, 409)
                    if not app.lock.acquire(blocking=False):
                        return self.reply({"error": "Дождитесь завершения предыдущего запроса"}, 409)
                    try:
                        allowed = {k: v for k, v in body.get("settings", {}).items() if k in {"top_k", "context_k", "similarity_threshold", "use_reranker", "source_id", "prompt_variant"}}
                        checked = config(allowed)
                        return self.reply(app.pipeline.ask(body.get("question"), {k: checked[k] for k in allowed}))
                    finally:
                        app.lock.release()
                return self.reply({"error": "Не найдено"}, 404)
            except (ValueError, TypeError, KeyError) as exc:
                self.reply({"error": str(exc)}, 400)
            except Exception as exc:
                self.reply({"error": str(exc)}, 500)
    return Handler


def serve(port=8000, open_browser=False):
    app = Application()
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    except OSError as exc:
        raise RuntimeError(f"Порт {port} занят. Закройте предыдущий запуск или используйте --port 8001") from exc
    url = f"http://localhost:{port}"
    print("SCADA RAG: " + url, flush=True)
    print("Оставьте это окно открытым. Для остановки нажмите Ctrl+C.", flush=True)
    if documents():
        app.prepare(refresh=False)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
