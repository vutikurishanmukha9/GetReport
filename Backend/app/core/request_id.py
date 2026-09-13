import re
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches a unique X-Request-ID header to every request/response cycle
    for end-to-end tracing and observability.
    Validates incoming request IDs to defeat CRLF and header injection (CWE-113).
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        raw_id = request.headers.get("X-Request-ID", "")
        if raw_id and SAFE_REQUEST_ID_REGEX.match(raw_id):
            request_id = raw_id
        else:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
