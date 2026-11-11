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

        path = scope.get("path", "")
        is_graph_html = "/graph/html" in path or path.endswith("/graph")

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))

                # Allow Vis.js CDN scripts & styles and framing from verified frontend origins
                if is_graph_html:
                    csp = (
                        b"default-src 'self'; "
                        b"script-src 'self' 'unsafe-inline' https://unpkg.com; "
                        b"style-src 'self' 'unsafe-inline' https://unpkg.com; "
                        b"font-src 'self' https://unpkg.com; "
                        b"object-src 'none'; "
                        b"frame-ancestors 'self' https://get-report.vercel.app https://getreport.vercel.app;"
                    )
                    frame_options = b"SAMEORIGIN"
                else:
                    csp = (
                        b"default-src 'self'; "
                        b"script-src 'self' 'unsafe-inline' https://unpkg.com; "
                        b"style-src 'self' 'unsafe-inline' https://unpkg.com; "
                        b"object-src 'none'; "
                        b"frame-ancestors 'none';"
                    )
                    frame_options = b"DENY"

                headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", frame_options),
                    (b"x-xss-protection", b"1; mode=block"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
                    (b"strict-transport-security", b"max-age=63072000; includeSubDomains; preload"),
                    (b"content-security-policy", csp),
                    (b"server", b"GetReport-Secure"),
                ])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
