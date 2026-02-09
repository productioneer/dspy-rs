/**
 * Sandboxed Python REPL runner using Pyodide (Python-in-WebAssembly).
 *
 * Launched by the host (TS or Rust SandboxedInterpreter) as:
 *   deno run --allow-read=<runner-dir> --node-modules-dir=auto sandbox-runner.ts
 *
 * Pyodide is loaded from the npm package (no network needed at runtime).
 *
 * Speaks the same JSON-RPC 2.0 protocol as repl-runner.py:
 *   Host -> Runner (stdin):  JSON-RPC requests (execute, register, ping, shutdown)
 *   Runner -> Host (stdout): JSON-RPC responses + tool_call requests
 *   Host -> Runner (stdin):  JSON-RPC responses to tool_call requests
 *
 * Isolation: Python code runs inside Pyodide's WASM sandbox.
 * No host filesystem, network, or process access from Python code.
 */

// deno-lint-ignore-file no-explicit-any

// Pyodide types (minimal, loaded dynamically)
interface PyodideInterface {
  runPython(code: string): any;
  runPythonAsync(code: string): Promise<any>;
  globals: PyProxy;
  setStdout(options: { batched: (text: string) => void }): void;
  setStderr(options: { batched: (text: string) => void }): void;
  isPyProxy(obj: unknown): boolean;
}

interface PyProxy {
  get(name: string): any;
  set(name: string, value: any): void;
  has(name: string): boolean;
  delete(name: string): void;
  toJs(options?: { dict_converter?: any }): any;
}

// JSON-RPC error codes (same as repl-runner.py)
const ERRORS: Record<string, number> = {
  SyntaxError: -32000,
  NameError: -32001,
  TypeError: -32002,
  ValueError: -32003,
  AttributeError: -32004,
  IndexError: -32005,
  KeyError: -32006,
  RuntimeError: -32007,
  CodeInterpreterError: -32008,
  Unknown: -32099,
};

// ============================================================================
// Globals
// ============================================================================

let pyodide: PyodideInterface;
let toolNames: string[] = [];
let outputFields: Array<{ name: string; type?: string }> = [];
let requestCounter = 0;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// ============================================================================
// Synchronous I/O — all communication uses sync reads/writes
// ============================================================================

function writeLine(obj: unknown): void {
  const line = JSON.stringify(obj) + "\n";
  Deno.stdout.writeSync(encoder.encode(line));
}

/** Read one line from stdin (blocking). Returns null on EOF. */
function readLineSync(): string | null {
  const buf = new Uint8Array(1);
  let line = "";

  while (true) {
    const n = Deno.stdin.readSync(buf);
    if (n === null) return line.length > 0 ? line : null; // EOF

    const ch = decoder.decode(buf.subarray(0, n));
    if (ch === "\n") {
      return line;
    }
    line += ch;
  }
}

// ============================================================================
// Pyodide initialization
// ============================================================================

async function initPyodide(): Promise<PyodideInterface> {
  const { loadPyodide } = await import("pyodide");

  const py = (await loadPyodide()) as PyodideInterface;

  // Block sandbox escape via `import js` which gives access to Deno APIs.
  // Remove the `js` module from Python's module registry so user code
  // cannot `import js` to reach globalThis.Deno.
  py.runPython(`
import sys as _sys
for _mod_name in list(_sys.modules.keys()):
    if _mod_name == 'js' or _mod_name.startswith('js.'):
        del _sys.modules[_mod_name]
# Also prevent re-importing by registering a broken finder
import importlib as _importlib
class _BlockJsImport:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name == 'js' or name.startswith('js.'):
            raise ImportError("'js' module is blocked in sandboxed mode")
        return None
    @classmethod
    def find_module(cls, name, path=None):
        if name == 'js' or name.startswith('js.'):
            raise ImportError("'js' module is blocked in sandboxed mode")
        return None
_sys.meta_path.insert(0, _BlockJsImport)
del _sys, _importlib, _BlockJsImport, _mod_name
`);

  return py;
}

// ============================================================================
// Python execution environment setup
// ============================================================================

/**
 * Install the execution wrapper in Pyodide. The wrapper:
 * - Catches _SubmitSignal and stores the result
 * - Captures stdout
 * - Returns a structured result dict
 * - Tool proxies call back into JS via _tool_call_bridge
 */
function setupPythonEnvironment(): void {
  pyodide.runPython(`
import io as _io
import json as _json
import sys as _sys

class _SubmitSignal(BaseException):
    """Raised by SUBMIT() to signal final output."""
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

# Persistent namespace for user code
_namespace = {"__builtins__": __builtins__}
_tool_names = []
_output_fields = []

def _make_tool_proxy(name):
    """Create a tool proxy that calls back to the JS host."""
    def proxy(**kwargs):
        result_json = _tool_call_bridge(name, _json.dumps(kwargs))
        result = _json.loads(result_json)
        value = result.get("value", "")
        if result.get("type") == "json":
            return _json.loads(value)
        return value
    proxy.__name__ = name
    proxy.__doc__ = f"Call host tool: {name}"
    return proxy

def _execute_code(code, variables_json):
    """Execute code in the persistent namespace. Returns a JSON-serializable result dict."""
    # Parse variables
    variables = _json.loads(variables_json) if variables_json else None

    # Inject variables
    if variables:
        for k, v in variables.items():
            _namespace[k] = v

    # Ensure SUBMIT and tools are in namespace
    _namespace["SUBMIT"] = _make_submit(_output_fields)
    for tname in _tool_names:
        _namespace[tname] = _make_tool_proxy(tname)

    # Capture stdout
    capture = _io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = capture

    try:
        exec(code, _namespace)
        output = capture.getvalue()
        _sys.stdout = old_stdout
        return _json.dumps({"type": "output", "output": output if output else None})
    except _SubmitSignal as s:
        _sys.stdout = old_stdout
        return _json.dumps({"type": "final", "value": s.value})
    except SyntaxError as e:
        _sys.stdout = old_stdout
        return _json.dumps({"type": "error", "error_type": "SyntaxError", "message": str(e)})
    except BaseException as e:
        _sys.stdout = old_stdout
        etype = type(e).__name__
        return _json.dumps({"type": "error", "error_type": etype, "message": str(e)})
    finally:
        _sys.stdout = old_stdout

def _register(tools_json, outputs_json):
    """Register tools and output fields."""
    global _tool_names, _output_fields
    _tool_names = _json.loads(tools_json) if tools_json else []
    _output_fields = _json.loads(outputs_json) if outputs_json else []
`);
}

// ============================================================================
// Tool call bridge
// ============================================================================

/**
 * JS function called by Python tool proxies.
 * Sends a tool_call JSON-RPC request to the host, reads the response synchronously.
 */
function toolCallBridge(name: string, kwargsJson: string): string {
  requestCounter++;
  const rid = `tool-${requestCounter}`;
  const request = {
    jsonrpc: "2.0",
    method: "tool_call",
    params: { name, kwargs: JSON.parse(kwargsJson) },
    id: rid,
  };
  writeLine(request);

  // Read response synchronously
  const line = readLineSync();
  if (!line) {
    throw new Error(`No response from host for tool call '${name}'`);
  }

  const resp = JSON.parse(line.trim());
  if (resp.error) {
    throw new Error(resp.error.message || "Tool call failed");
  }

  const result = resp.result || {};
  // Return JSON with both value and type so Python proxy can decide whether to parse
  return JSON.stringify({ value: result.value || "", type: result.type || "string" });
}

// ============================================================================
// Code execution
// ============================================================================

interface JsonRpcMessage {
  jsonrpc: string;
  method?: string;
  params?: Record<string, unknown>;
  id?: string | number | null;
}

interface JsonRpcResponse {
  jsonrpc: string;
  result?: Record<string, unknown>;
  error?: { code: number; message: string; data?: Record<string, unknown> };
  id: string | number | null;
}

function executeCode(
  code: string,
  variables: Record<string, unknown> | undefined,
  msgId: string | number | null,
): JsonRpcResponse {
  const variablesJson = variables ? JSON.stringify(variables) : "";

  // Call the Python execution wrapper
  const resultJson = pyodide.runPython(
    `_execute_code(${JSON.stringify(code)}, ${JSON.stringify(variablesJson)})`,
  ) as string;

  const result = JSON.parse(resultJson);

  switch (result.type) {
    case "output":
      return {
        jsonrpc: "2.0",
        result: { output: result.output },
        id: msgId,
      };

    case "final":
      return {
        jsonrpc: "2.0",
        result: { final: result.value },
        id: msgId,
      };

    case "error": {
      const errorType = result.error_type || "Unknown";
      const errorMessage = result.message || "Unknown error";
      const codeNum = ERRORS[errorType] ?? ERRORS.Unknown;
      return {
        jsonrpc: "2.0",
        error: {
          code: codeNum,
          message: errorMessage,
          data: { type: errorType, args: errorMessage },
        },
        id: msgId,
      };
    }

    default:
      return {
        jsonrpc: "2.0",
        error: {
          code: ERRORS.Unknown,
          message: "Unexpected execution result type",
        },
        id: msgId,
      };
  }
}

// ============================================================================
// Message handler
// ============================================================================

function handleMessage(msg: JsonRpcMessage): JsonRpcResponse | null {
  const method = msg.method;
  const params = msg.params || {};
  const msgId = msg.id ?? null;

  switch (method) {
    case "execute":
      return executeCode(
        (params.code as string) || "",
        params.variables as Record<string, unknown> | undefined,
        msgId,
      );

    case "register": {
      const tools = (params.tools as Array<{ name: string }>) || [];
      const toolNamesJson = JSON.stringify(tools.map((t) => t.name));
      const outputsJson = JSON.stringify(params.outputs || []);
      // Call Python _register directly via globals to avoid string interpolation
      const registerFn = pyodide.globals.get("_register");
      registerFn(toolNamesJson, outputsJson);
      // Also update JS-side tracking
      toolNames = tools.map((t) => t.name);
      outputFields =
        (params.outputs as Array<{ name: string; type?: string }>) || [];
      return { jsonrpc: "2.0", result: { ok: true }, id: msgId };
    }

    case "ping":
      return { jsonrpc: "2.0", result: { ok: true }, id: msgId };

    case "shutdown":
      if (msgId !== null && msgId !== undefined) {
        writeLine({ jsonrpc: "2.0", result: { ok: true }, id: msgId });
      }
      Deno.exit(0);
      return null; // unreachable

    default:
      return {
        jsonrpc: "2.0",
        error: { code: -32601, message: `Method not found: ${method}` },
        id: msgId,
      };
  }
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  // Initialize Pyodide (this is the slow part — downloads WASM from CDN)
  pyodide = await initPyodide();
  setupPythonEnvironment();

  // Register the tool call bridge in Python's namespace
  pyodide.globals.set("_tool_call_bridge", toolCallBridge);

  // Process JSON-RPC messages (synchronous main loop)
  while (true) {
    const line = readLineSync();
    if (line === null) break; // EOF

    const trimmed = line.trim();
    if (!trimmed) continue;

    let msg: JsonRpcMessage;
    try {
      msg = JSON.parse(trimmed);
    } catch {
      writeLine({
        jsonrpc: "2.0",
        error: { code: -32700, message: "Parse error" },
        id: null,
      });
      continue;
    }

    const response = handleMessage(msg);
    if (response !== null) {
      writeLine(response);
    }
  }
}

main().catch((err) => {
  console.error("Sandbox runner fatal error:", err);
  Deno.exit(1);
});
