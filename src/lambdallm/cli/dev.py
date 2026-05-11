"""lambdallm dev - Local development server.

Emulates Lambda execution locally with hot-reload support.
No AWS credentials needed for basic development.
"""

import json
import importlib
import sys
import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger("lambdallm")


class MockLambdaContext:
    """Mock Lambda context for local development."""

    function_name = "local-dev"
    function_version = "$LATEST"
    memory_limit_in_mb = 256
    invoked_function_arn = "arn:aws:lambda:local:000000000:function:local-dev"
    log_group_name = "/aws/lambda/local-dev"
    log_stream_name = "local"
    _timeout_ms = 900_000  # 15 minutes

    def get_remaining_time_in_millis(self):
        return self._timeout_ms


class DevRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that routes requests to Lambda handlers."""

    handler_func = None

    def do_POST(self):
        """Handle POST requests (main invocation path)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"

        # Build Lambda-like event
        event = {
            "httpMethod": "POST",
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
            "requestContext": {"requestId": "local-dev-request"},
        }

        # Invoke the handler
        try:
            result = self.handler_func(event, MockLambdaContext())

            status_code = result.get("statusCode", 200) if isinstance(result, dict) else 200
            response_body = result.get("body", "{}") if isinstance(result, dict) else json.dumps(result)

            if isinstance(response_body, dict):
                response_body = json.dumps(response_body)

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response_body.encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error_body = json.dumps({"error": str(e), "type": type(e).__name__})
            self.wfile.write(error_body.encode("utf-8"))

    def do_GET(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "running",
            "framework": "lambdallm",
            "mode": "development",
        }).encode("utf-8"))

    def log_message(self, format, *args):
        """Custom log format."""
        logger.info(f"[DEV] {args[0]} {args[1]} {args[2]}")


def start_dev_server(port: int = 3000, handler: Optional[str] = None):
    """Start the local development server.

    Args:
        port: Port to listen on (default: 3000).
        handler: Specific handler module to serve (e.g., "handlers.main").
    """
    # Add current directory to path for handler imports
    sys.path.insert(0, os.getcwd())

    # Find and load the handler
    handler_func = _load_handler(handler)
    if not handler_func:
        print("Error: No handler found. Make sure you have handlers/main.py")
        print("Or specify: lambdallm dev --handler handlers.main")
        sys.exit(1)

    DevRequestHandler.handler_func = handler_func

    server = HTTPServer(("0.0.0.0", port), DevRequestHandler)

    print(f"LambdaLLM dev server running on http://localhost:{port}")
    print(f"Handler: {handler or 'handlers.main'}")
    print(f"Press Ctrl+C to stop")
    print()
    print(f"Test with:")
    print(f'  curl -X POST http://localhost:{port} -d \'{{"text": "Hello world"}}\'')
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dev server.")
        server.shutdown()


def invoke_handler(handler_name: str, data: Optional[str] = None, file: Optional[str] = None):
    """Invoke a handler locally with provided data."""
    sys.path.insert(0, os.getcwd())

    handler_func = _load_handler(f"handlers.{handler_name}")
    if not handler_func:
        print(f"Error: Handler '{handler_name}' not found.")
        sys.exit(1)

    # Load input data
    if file:
        with open(file, "r") as f:
            body = f.read()
    elif data:
        body = data
    else:
        body = "{}"

    event = {"httpMethod": "POST", "path": "/", "body": body, "headers": {}, "requestContext": {}}

    result = handler_func(event, MockLambdaContext())

    print(json.dumps(result, indent=2))


def _load_handler(module_path: Optional[str] = None):
    """Load a handler function from a module path."""
    paths_to_try = [module_path] if module_path else ["handlers.main", "handler", "main"]

    for path in paths_to_try:
        if not path:
            continue
        try:
            module = importlib.import_module(path)
            # Look for lambda_handler or handler function
            for attr_name in ["lambda_handler", "handler", "main"]:
                func = getattr(module, attr_name, None)
                if func and callable(func):
                    return func
        except (ImportError, ModuleNotFoundError):
            continue

    return None
