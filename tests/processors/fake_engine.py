"""A minimal OpenAI-compatible chat endpoint on localhost, for tests.

Not named test_* so pytest does not collect it.

The batch tests mock LLMClient, so they cannot see what actually goes on the
wire. This serves /v1/chat/completions over real HTTP, records every request
body it receives, and answers the way vLLM does: each choice carries its own
finish_reason, "length" when max_tokens cut the reply short.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: The reply the fake engine "generates", one token per word.
REPLY_WORDS = "the quick brown fox jumps over the lazy dog".split()


class FakeEngine:
    """Context manager: `with FakeEngine() as engine: engine.url`."""

    def __init__(self, extra_response_fields=None):
        self.requests = []
        self.extra_response_fields = extra_response_fields or {}
        self._lock = threading.Lock()

    def __enter__(self):
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with engine._lock:
                    engine.requests.append(body)
                status = engine.status(body)
                reply = engine.respond(body)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def status(self, body):
        """HTTP status for this request; override with respond() to fail one."""
        return 200

    def respond(self, body):
        limit = body.get("max_tokens") or len(REPLY_WORDS)
        words = REPLY_WORDS[:limit]
        finish = "length" if limit < len(REPLY_WORDS) else "stop"
        choices = [
            {
                "index": i,
                "message": {"role": "assistant", "content": " ".join(words)},
                "logprobs": None,
                "finish_reason": finish,
                "stop_reason": None,
            }
            for i in range(body.get("n") or 1)
        ]
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body["model"],
            "choices": choices,
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": len(words) * len(choices),
                "total_tokens": 3 + len(words) * len(choices),
            },
            **self.extra_response_fields,
        }
