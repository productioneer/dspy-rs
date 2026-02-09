#!/usr/bin/env python3
"""
Unsandboxed Python REPL runner with JSON-RPC 2.0 protocol over stdin/stdout.

Used by the DSPy TS/Rust code interpreter to execute Python code.
State persists across execute() calls within a session.

Protocol:
  Host -> Runner (stdin):  JSON-RPC requests (execute, register, shutdown)
  Runner -> Host (stdout): JSON-RPC responses + tool_call requests
  Host -> Runner (stdin):  JSON-RPC responses to tool_call requests
"""

import io
import json
import sys
import traceback

# ============================================================================
# JSON-RPC Error Codes
# ============================================================================

ERRORS = {
    "SyntaxError": -32000,
    "NameError": -32001,
    "TypeError": -32002,
    "ValueError": -32003,
    "AttributeError": -32004,
    "IndexError": -32005,
    "KeyError": -32006,
    "RuntimeError": -32007,
    "CodeInterpreterError": -32008,
    "Unknown": -32099,
}

# ============================================================================
# Globals
# ============================================================================

# Save real stdout/stdin before anything can redirect them
_real_stdout = sys.stdout
_real_stdin = sys.stdin

namespace = {"__builtins__": __builtins__}
tool_names = []
output_fields = []
request_counter = 0


# ============================================================================
# SUBMIT mechanism
# ============================================================================

class _SubmitSignal(BaseException):
    """Raised by SUBMIT() to signal final output. Uses BaseException to avoid
    being caught by except Exception."""
    def __init__(self, value):
        self.value = value


def _make_submit(fields):
    """Create the SUBMIT function based on registered output fields."""
    if fields:
        field_names = [f["name"] for f in fields]
        def SUBMIT(**kwargs):
            missing = set(field_names) - set(kwargs.keys())
            if missing:
                raise ValueError(f"SUBMIT missing fields: {sorted(missing)}")
            raise _SubmitSignal(kwargs)
    else:
        def SUBMIT(*args, **kwargs):
            if kwargs and not args:
                raise _SubmitSignal(kwargs)
            elif args and not kwargs:
                raise _SubmitSignal(args[0] if len(args) == 1 else list(args))
            elif args and kwargs:
                raise _SubmitSignal({"positional": list(args), **kwargs})
            else:
                raise _SubmitSignal(None)
    SUBMIT.__doc__ = "Submit final output and end execution."
    return SUBMIT


# ============================================================================
# Tool proxies — always use _real_stdout/_real_stdin for JSON-RPC
# ============================================================================

def _make_tool_proxy(name):
    """Create a proxy function that calls a tool on the host via JSON-RPC.
    Uses _real_stdout/_real_stdin so tool calls work even when sys.stdout
    is redirected for print() capture."""
    def proxy(**kwargs):
        global request_counter
        request_counter += 1
        rid = f"tool-{request_counter}"
        request = {
            "jsonrpc": "2.0",
            "method": "tool_call",
            "params": {"name": name, "kwargs": kwargs},
            "id": rid,
        }
        # Send to host via REAL stdout (not captured stdout)
        _real_stdout.write(json.dumps(request) + "\n")
        _real_stdout.flush()
        # Read response from host via REAL stdin
        line = _real_stdin.readline().strip()
        if not line:
            raise RuntimeError(f"No response from host for tool call '{name}'")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(resp["error"].get("message", "Tool call failed"))
        result = resp.get("result", {})
        value = result.get("value", "")
        if result.get("type") == "json":
            return json.loads(value)
        return value
    proxy.__name__ = name
    proxy.__doc__ = f"Call host tool: {name}"
    return proxy


# ============================================================================
# Code execution
# ============================================================================

def execute_code(code, variables, msg_id):
    """Execute Python code in the persistent namespace."""
    # Inject variables
    if variables:
        for k, v in variables.items():
            namespace[k] = v

    # Ensure SUBMIT and tools are in namespace
    namespace["SUBMIT"] = _make_submit(output_fields)
    for tname in tool_names:
        namespace[tname] = _make_tool_proxy(tname)

    # Capture stdout — redirect sys.stdout so print() goes to capture buffer
    # Tool proxies use _real_stdout directly, so they bypass this.
    capture = io.StringIO()
    sys.stdout = capture

    try:
        exec(code, namespace)
        output = capture.getvalue()
        sys.stdout = _real_stdout
        return {"jsonrpc": "2.0", "result": {"output": output}, "id": msg_id}
    except _SubmitSignal as s:
        sys.stdout = _real_stdout
        return {"jsonrpc": "2.0", "result": {"final": s.value}, "id": msg_id}
    except SyntaxError as e:
        sys.stdout = _real_stdout
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": ERRORS["SyntaxError"],
                "message": str(e),
                "data": {"type": "SyntaxError", "args": str(e)},
            },
            "id": msg_id,
        }
    except BaseException as e:
        sys.stdout = _real_stdout
        etype = type(e).__name__
        code_num = ERRORS.get(etype, ERRORS["Unknown"])
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code_num,
                "message": str(e),
                "data": {"type": etype, "args": str(e)},
            },
            "id": msg_id,
        }
    finally:
        sys.stdout = _real_stdout


# ============================================================================
# Message handler
# ============================================================================

def handle_message(msg):
    global tool_names, output_fields

    method = msg.get("method")
    params = msg.get("params", {})
    msg_id = msg.get("id")

    if method == "execute":
        return execute_code(
            params.get("code", ""),
            params.get("variables"),
            msg_id,
        )

    elif method == "register":
        tools_info = params.get("tools", [])
        tool_names = [t["name"] for t in tools_info]
        output_fields = params.get("outputs", [])
        return {"jsonrpc": "2.0", "result": {"ok": True}, "id": msg_id}

    elif method == "ping":
        return {"jsonrpc": "2.0", "result": {"ok": True}, "id": msg_id}

    elif method == "shutdown":
        # Send response before exiting
        if msg_id is not None:
            resp = {"jsonrpc": "2.0", "result": {"ok": True}, "id": msg_id}
            _real_stdout.write(json.dumps(resp) + "\n")
            _real_stdout.flush()
        sys.exit(0)

    else:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
            "id": msg_id,
        }


# ============================================================================
# Main loop
# ============================================================================

def main():
    for line in _real_stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None,
            }
            _real_stdout.write(json.dumps(resp) + "\n")
            _real_stdout.flush()
            continue

        response = handle_message(msg)
        if response is not None:
            _real_stdout.write(json.dumps(response) + "\n")
            _real_stdout.flush()


if __name__ == "__main__":
    main()
