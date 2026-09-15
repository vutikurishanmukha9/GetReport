from starlette.types import ASGIApp, Scope, Receive, Send


class SecurityHeadersMiddleware:
    """
    Pure ASGI Middleware that adds enterprise security headers to every HTTP response.
    Replaces BaseHTTPMiddleware to support StreamingResponse (SSE) without deadlocks or buffer bloat.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"deny"),
                    (b"x-xss-protection", b"1; mode=block"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
                    (b"strict-transport-security", b"max-age=63072000; includeSubDomains; preload"),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none';"),
                    (b"server", b"GetReport-Secure"),
                ])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
