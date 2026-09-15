import re
import uuid
from starlette.types import ASGIApp, Scope, Receive, Send

SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class RequestIDMiddleware:
    """
    Pure ASGI Middleware that attaches a unique X-Request-ID header to every request/response cycle.
    Replaces BaseHTTPMiddleware to support StreamingResponse (SSE) without deadlocks or buffer bloat.
    Validates incoming request IDs to defeat CRLF and header injection (CWE-113).
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        req_headers = dict(scope.get("headers", []))
        raw_id = req_headers.get(b"x-request-id", b"").decode("latin-1")
        if raw_id and SAFE_REQUEST_ID_REGEX.match(raw_id):
            request_id = raw_id
        else:
            request_id = str(uuid.uuid4())

        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)
