from __future__ import annotations

import json
import socket
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final


@dataclass(frozen=True, slots=True)
class EmulatedOpenAIResponse:
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    delay_seconds: float = 0
    close_connection: bool = False


@dataclass(frozen=True, slots=True)
class EmulatedOpenAIRequest:
    profile: str
    path: str
    model: str | None
    streaming: bool


class OpenAISubscriptionEmulator:
    """A local, stateful HTTP upstream for subscription-pool integration tests."""

    def __init__(self, profile_tokens: Mapping[str, str]) -> None:
        self._profile_by_token: Final = {token: profile for profile, token in profile_tokens.items()}
        if len(self._profile_by_token) != len(profile_tokens):
            raise ValueError("emulator profile tokens must be unique")
        self._responses: defaultdict[str, deque[EmulatedOpenAIResponse]] = defaultdict(deque)
        self._requests: list[EmulatedOpenAIRequest] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        server = self._server
        if server is None:
            raise RuntimeError("emulator is not running")
        host, port = server.server_address
        return f"http://{host}:{port}"

    @property
    def requests(self) -> tuple[EmulatedOpenAIRequest, ...]:
        with self._lock:
            return tuple(self._requests)

    def enqueue(self, profile: str, *responses: EmulatedOpenAIResponse) -> None:
        if profile not in self._profile_by_token.values():
            raise ValueError("unknown emulator profile")
        with self._lock:
            self._responses[profile].extend(responses)

    def start(self) -> OpenAISubscriptionEmulator:
        if self._server is not None:
            raise RuntimeError("emulator is already running")
        emulator = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                emulator._handle_request(self)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> OpenAISubscriptionEmulator:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def _handle_request(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path not in {"/chat/completions", "/responses"}:
            self._write_json(handler, 404, {"error": {"message": "unknown emulator endpoint"}})
            return

        profile = self._authenticate(handler.headers.get("Authorization"))
        if profile is None:
            self._write_json(handler, 401, {"error": {"message": "emulated authentication failure"}})
            return

        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = -1
        if not 0 <= content_length <= 1024 * 1024:
            self._write_json(handler, 400, {"error": {"message": "invalid emulator request size"}})
            return
        try:
            payload = json.loads(handler.rfile.read(content_length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._write_json(handler, 400, {"error": {"message": "invalid emulator JSON"}})
            return
        if not isinstance(payload, dict):
            self._write_json(handler, 400, {"error": {"message": "invalid emulator request"}})
            return

        with self._lock:
            self._requests.append(
                EmulatedOpenAIRequest(
                    profile=profile,
                    path=handler.path,
                    model=payload.get("model") if isinstance(payload.get("model"), str) else None,
                    streaming=payload.get("stream") is True,
                )
            )
            response = self._responses[profile].popleft() if self._responses[profile] else EmulatedOpenAIResponse()

        if response.delay_seconds > 0:
            time.sleep(response.delay_seconds)
        if response.close_connection:
            try:
                handler.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            handler.connection.close()
            return
        if response.status_code != 200:
            self._write_json(
                handler,
                response.status_code,
                {
                    "error": {
                        "message": "emulated provider failure",
                        "type": "emulated_error",
                        "code": "emulated_error",
                    }
                },
                response.headers,
            )
            return

        if handler.path == "/responses":
            self._write_sse(handler, self._responses_api_events(payload), response.headers)
        elif payload.get("stream") is True:
            self._write_sse(handler, self._chat_completion_events(payload), response.headers)
        else:
            self._write_json(handler, 200, self._chat_completion(payload), response.headers)

    def _authenticate(self, authorization: str | None) -> str | None:
        if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
            return None
        return self._profile_by_token.get(authorization.removeprefix("Bearer "))

    @staticmethod
    def _chat_completion(payload: Mapping[str, object]) -> dict[str, object]:
        model = payload.get("model") if isinstance(payload.get("model"), str) else "gpt-5.4"
        return {
            "id": "chatcmpl_emulated",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "emulated response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @staticmethod
    def _chat_completion_events(payload: Mapping[str, object]) -> tuple[dict[str, object], ...]:
        model = payload.get("model") if isinstance(payload.get("model"), str) else "gpt-5.4"
        common = {
            "id": "chatcmpl_emulated_stream",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
        }
        return (
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "emulated "},
                        "finish_reason": None,
                    }
                ],
            },
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "stream"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    @staticmethod
    def _responses_api_events(payload: Mapping[str, object]) -> tuple[dict[str, object], ...]:
        model = payload.get("model") if isinstance(payload.get("model"), str) else "gpt-5.4"
        response = {
            "id": "resp_emulated",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": model,
            "output": [
                {
                    "id": "msg_emulated",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "emulated response",
                            "annotations": [],
                        }
                    ],
                }
            ],
        }
        return (
            {
                "type": "response.output_text.delta",
                "item_id": "msg_emulated",
                "output_index": 0,
                "content_index": 0,
                "delta": "emulated ",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_emulated",
                "output_index": 0,
                "content_index": 0,
                "delta": "stream",
            },
            {"type": "response.completed", "response": response},
        )

    @staticmethod
    def _write_json(
        handler: BaseHTTPRequestHandler,
        status_code: int,
        payload: Mapping[str, object],
        headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        OpenAISubscriptionEmulator._write_response(
            handler,
            status_code,
            body,
            {"Content-Type": "application/json", **dict(headers or {})},
        )

    @staticmethod
    def _write_sse(
        handler: BaseHTTPRequestHandler,
        events: Iterable[Mapping[str, object]],
        headers: Mapping[str, str] | None = None,
    ) -> None:
        lines = [f"data: {json.dumps(event, separators=(',', ':'))}\n\n" for event in events]
        lines.append("data: [DONE]\n\n")
        body = "".join(lines).encode("utf-8")
        OpenAISubscriptionEmulator._write_response(
            handler,
            200,
            body,
            {"Content-Type": "text/event-stream", **dict(headers or {})},
        )

    @staticmethod
    def _write_response(
        handler: BaseHTTPRequestHandler,
        status_code: int,
        body: bytes,
        headers: Mapping[str, str],
    ) -> None:
        try:
            handler.send_response(status_code)
            for name, value in headers.items():
                handler.send_header(name, value)
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
