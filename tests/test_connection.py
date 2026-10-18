import json
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from scada_rag.generation.ollama import request_json
from scada_rag.server import Application, make_handler


@contextmanager
def running_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class ConnectionTests(unittest.TestCase):
    def test_loopback_ignores_broken_system_proxy(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                data = b'{"models": []}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        proxy = urllib.request.build_opener(urllib.request.ProxyHandler({"http": "http://127.0.0.1:1"}))
        with running_server(Handler) as base, patch("urllib.request._opener", proxy), patch("urllib.request.proxy_bypass", return_value=False):
            self.assertEqual(request_json(base, "/api/tags", timeout=1), {"models": []})

    def test_missing_model_gives_exact_pull_command(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error": "model qwen2.5:3b not found"}')

            def log_message(self, *_):
                pass

        with running_server(Handler) as base:
            with self.assertRaisesRegex(RuntimeError, "ollama pull qwen2.5:3b"):
                request_json(base, "/api/chat", {"model": "qwen2.5:3b"}, timeout=1)

    def test_slow_ollama_returns_timeout_error(self):
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                release.wait(timeout=2)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_):
                pass

        with running_server(Handler) as base:
            try:
                with self.assertRaisesRegex(RuntimeError, "не завершила запрос"):
                    request_json(base, "/slow", timeout=0.05)
            finally:
                release.set()

    def test_invalid_ollama_json_is_reported(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'<html>proxy page</html>')

            def log_message(self, *_):
                pass

        with running_server(Handler) as base:
            with self.assertRaisesRegex(RuntimeError, "некорректный ответ"):
                request_json(base, "/api/tags", timeout=1)

    def test_status_failure_returns_json_error(self):
        class App:
            def status(self):
                raise ValueError("Повреждён файл состояния")

        with running_server(make_handler(App())) as base, patch("scada_rag.server.traceback.print_exc"):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + "/api/status", timeout=1)
            self.assertEqual(error.exception.code, 500)
            self.assertIn("Повреждён файл", json.load(error.exception)["error"])

    def test_status_responds_during_generation_and_lock_is_released(self):
        started, release = threading.Event(), threading.Event()
        results = []

        class SlowPipeline:
            store = type("Store", (), {"meta": {"chunk_count": 10}})()

            def ask(self, *_):
                started.set()
                release.wait(timeout=3)
                return {"answer": "Готово"}

        app = Application()
        app.pipeline = SlowPipeline()
        app.state["ready"] = True
        with running_server(make_handler(app)) as base, patch("scada_rag.server.documents", return_value=[]), patch("scada_rag.server.request_json", return_value={"models": [{"name": "qwen2.5:3b"}]}), patch.object(app, "log"):
            def ask():
                try:
                    request = urllib.request.Request(base + "/api/ask", data=b'{"question":"test"}', headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=3) as response:
                        results.append(json.load(response))
                except Exception as exc:
                    results.append(exc)

            worker = threading.Thread(target=ask, daemon=True)
            worker.start()
            try:
                self.assertTrue(started.wait(timeout=2))
                with urllib.request.urlopen(base + "/api/status", timeout=1) as response:
                    self.assertTrue(json.load(response)["answering"])
            finally:
                release.set()
                worker.join(timeout=3)
            self.assertEqual(results, [{"answer": "Готово"}])
            self.assertFalse(app.state["answering"])
            self.assertTrue(app.lock.acquire(blocking=False))
            app.lock.release()

    def test_preparation_waits_for_reranker(self):
        started, release = threading.Event(), threading.Event()

        class FakePipeline:
            cfg = {"use_reranker": True}

            def __init__(self, **_):
                pass

            def load_reranker(self, _):
                started.set()
                release.wait(timeout=3)

        app = Application()
        with patch("scada_rag.pipeline.Pipeline", FakePipeline), patch.object(app, "log"):
            app.prepare(refresh=False)
            try:
                self.assertTrue(started.wait(timeout=2))
                self.assertFalse(app.state["ready"])
                self.assertTrue(app.state["busy"])
            finally:
                release.set()
            self.assertTrue(app.lock.acquire(timeout=3))
            try:
                self.assertTrue(app.state["ready"])
                self.assertFalse(app.state["busy"])
            finally:
                app.lock.release()


if __name__ == "__main__":
    unittest.main()
