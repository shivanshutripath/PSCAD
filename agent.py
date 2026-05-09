"""
VeraGrid Agent Loop
===================
Implements the think → act → observe → repeat cycle for OpenAI, Claude,
and Cursor models with real tool use.

For OpenAI / Claude we drive the agent loop ourselves using their native
tool-use APIs. For Cursor we shell out to the `cursor-agent` CLI in
`--print` mode, which itself is an agent loop with built-in tools (file
read/write, shell, search, ...). We parse its NDJSON event stream to
build a comparable trace.

The agent receives a question, decides which tools to call,
executes them, observes results, and continues until it has
an answer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from tools import TOOL_SCHEMAS, dispatch_tool
from prompt_builder import build_full_system_prompt


def _build_trace(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "used_tools": False,
        "tool_call_count": 0,
        "wrote_files": False,
        "simulation_executed": False,
        "read_files": False,
        "listed_files": False,
        "tool_calls": [],
        "errors": [],
    }


def _append_tool_trace(trace: dict[str, Any], tool_name: str, tool_args: dict[str, Any], result: Any) -> None:
    trace["used_tools"] = True
    trace["tool_call_count"] += 1
    if tool_name == "write_file":
        trace["wrote_files"] = True
    elif tool_name in {"execute_veragrid", "run_web_opf"}:
        trace["simulation_executed"] = True
    elif tool_name == "read_file":
        trace["read_files"] = True
    elif tool_name == "list_files":
        trace["listed_files"] = True

    content_preview = str(getattr(result, "content", ""))[:500]
    call_record: dict[str, Any] = {
        "name": tool_name,
        "args": tool_args,
        "success": bool(getattr(result, "success", False)),
        "content_preview": content_preview,
    }

    if tool_name in {"execute_veragrid", "run_web_opf"}:
        try:
            meta = json.loads(getattr(result, "content", "") or "{}")
            call_record["return_code"] = meta.get("return_code")
            call_record["output_files"] = meta.get("output_files", [])
            if not call_record["output_files"] and meta.get("results_file"):
                call_record["output_files"] = [meta.get("results_file")]
        except Exception:
            pass

    trace["tool_calls"].append(call_record)

    if not call_record["success"]:
        trace["errors"].append(f"{tool_name} failed")


# ---------------------------------------------------------------------------
# OpenAI Agent (uses function calling with tool loop)
# ---------------------------------------------------------------------------

def is_openai_reasoning_model(model: str) -> bool:
    """o-series and gpt-5* are reasoning families: hidden reasoning tokens
    count toward the output budget, and they reject `temperature` other than 1
    on chat.completions. They also expect `max_completion_tokens` instead of
    the legacy `max_tokens` field."""
    m = (model or "").lower().strip()
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m.startswith("gpt-5"):
        return True
    return False


def run_openai_agent(
    model: str,
    user_prompt: str,
    workspace: str,
    api_key: str | None = None,
    circuit_data: dict[str, Any] | None = None,
    answer_format: str = "letter_only",
    return_trace: bool = False,
) -> str | dict[str, Any]:
    """
    Run the full agent loop using OpenAI's function-calling API.

    Flow:
    1. Send the question + tool schemas to the model
    2. If the model wants to call a tool → execute it, append result, loop
    3. If the model responds with text → return it as the answer
    """
    import openai

    key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key required for agent mode.")
    client = openai.OpenAI(api_key=key)

    # Build system prompt dynamically from circuit data
    system_prompt = build_full_system_prompt(
        circuit_data=circuit_data,
        answer_format=answer_format,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    max_iterations = 10
    trace = _build_trace(model)
    reasoning = is_openai_reasoning_model(model)

    for iteration in range(max_iterations):
        # Reasoning families (gpt-5*, o*) need `max_completion_tokens` with a
        # generous budget — hidden reasoning tokens are billed against it.
        # Older chat models still use the legacy `max_tokens` field. We don't
        # set `temperature` at all and let the model use its default.
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
        }
        if reasoning:
            create_kwargs["max_completion_tokens"] = 16384
        else:
            create_kwargs["max_tokens"] = 2048

        response = client.chat.completions.create(**create_kwargs)

        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", None)

        # If the model wants to call tools
        if message.tool_calls:
            # Append the assistant message (with tool calls) to history
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            # Execute each tool call and append results
            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                print(f"    [Tool Call {iteration+1}] {tool_name}({json.dumps(tool_args)[:100]}...)")
                result = dispatch_tool(workspace, tool_name, tool_args)
                _append_tool_trace(trace, tool_name, tool_args, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.content,
                })
            continue

        # Plain text reply → answer is ready
        if message.content:
            final_text = message.content.strip()
            if return_trace:
                return {"answer": final_text, "trace": trace}
            return final_text

        # No tool_calls AND empty content. This usually means the reasoning
        # model exhausted its output budget on hidden reasoning tokens
        # (`finish_reason == "length"`). Surface a clear error instead of
        # spinning forever inside the loop with the same empty state.
        if finish_reason == "length":
            error_text = (
                f"ERROR: model {model} returned empty content with "
                f"finish_reason=length (likely ran out of reasoning budget)."
            )
            trace["errors"].append(error_text)
            if return_trace:
                return {"answer": error_text, "trace": trace}
            return error_text

        if finish_reason == "stop":
            # Empty stop — return the empty string and let the caller's
            # extract_letter mark it wrong, rather than silently looping.
            if return_trace:
                return {"answer": "", "trace": trace}
            return ""

        # Any other unexpected state: bail out with a descriptive error.
        error_text = (
            f"ERROR: unexpected response from {model} "
            f"(finish_reason={finish_reason!r}, content empty, no tool_calls)."
        )
        trace["errors"].append(error_text)
        if return_trace:
            return {"answer": error_text, "trace": trace}
        return error_text

    error_text = "ERROR: Agent exceeded maximum iterations"
    trace["errors"].append(error_text)
    if return_trace:
        return {"answer": error_text, "trace": trace}
    return error_text


# ---------------------------------------------------------------------------
# Claude Agent (uses Anthropic tool_use API)
# ---------------------------------------------------------------------------

def run_claude_agent(
    model: str,
    user_prompt: str,
    workspace: str,
    api_key: str | None = None,
    circuit_data: dict[str, Any] | None = None,
    answer_format: str = "letter_only",
    return_trace: bool = False,
) -> str | dict[str, Any]:
    """
    Run the full agent loop using Anthropic's tool-use API.

    Flow is the same: send question + tools → handle tool_use blocks →
    return tool results → repeat until text response.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Install anthropic: pip install anthropic")

    key = (api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Anthropic API key required for agent mode.")
    client = anthropic.Anthropic(api_key=key)

    # Build system prompt dynamically from circuit data
    system_prompt = build_full_system_prompt(
        circuit_data=circuit_data,
        answer_format=answer_format,
    )

    # Convert tool schemas to Anthropic format
    claude_tools = []
    for schema in TOOL_SCHEMAS:
        func = schema["function"]
        claude_tools.append({
            "name": func["name"],
            "description": func["description"],
            "input_schema": func["parameters"],
        })

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]

    max_iterations = 10
    trace = _build_trace(model)

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=1024,
            tools=claude_tools,
            messages=messages,
        )

        # Check if there are any tool_use blocks
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if tool_use_blocks:
            # Append assistant response to history
            messages.append({
                "role": "assistant",
                "content": [
                    _serialize_block(b) for b in response.content
                ],
            })

            # Execute each tool and append results
            tool_results = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool_args = block.input or {}

                print(f"    [Tool Call {iteration+1}] {tool_name}({json.dumps(tool_args)[:100]}...)")
                result = dispatch_tool(workspace, tool_name, tool_args)
                _append_tool_trace(trace, tool_name, tool_args, result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.content,
                })

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn" or text_blocks:
            # Model is done — extract text
            texts = [b.text for b in text_blocks if hasattr(b, "text")]
            final_text = "\n".join(texts).strip()
            if return_trace:
                return {"answer": final_text, "trace": trace}
            return final_text

        else:
            error_text = "ERROR: Unexpected response from Claude agent"
            trace["errors"].append(error_text)
            if return_trace:
                return {"answer": error_text, "trace": trace}
            return error_text

    error_text = "ERROR: Agent exceeded maximum iterations"
    trace["errors"].append(error_text)
    if return_trace:
        return {"answer": error_text, "trace": trace}
    return error_text


def _serialize_block(block: Any) -> dict:
    """Convert an Anthropic content block to a serializable dict."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    return {"type": "text", "text": str(block)}


# ---------------------------------------------------------------------------
# Cursor Agent (shells out to the `cursor-agent` CLI)
# ---------------------------------------------------------------------------

# Common locations the official installer drops `cursor-agent` into.
_CURSOR_AGENT_FALLBACK_PATHS = (
    os.path.expanduser("~/.local/bin/cursor-agent"),
    os.path.expanduser("~/.cursor/cli/cursor-agent"),
    "/usr/local/bin/cursor-agent",
    "/opt/homebrew/bin/cursor-agent",
)


def _find_cursor_agent_binary() -> str | None:
    found = shutil.which("cursor-agent")
    if found:
        return found
    for candidate in _CURSOR_AGENT_FALLBACK_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _strip_cursor_prefix(model: str) -> str:
    """Frontend uses `cursor:` prefix to disambiguate; the CLI wants the bare name."""
    raw = (model or "").strip()
    if raw.lower().startswith("cursor:"):
        return raw.split(":", 1)[1].strip() or "auto"
    if raw.lower().startswith("cursor-"):
        return raw[len("cursor-"):].strip() or "auto"
    return raw or "auto"


def _tool_name_from_event(tool_call_obj: dict[str, Any]) -> str:
    """The CLI uses one wrapper key per tool kind (readToolCall, writeToolCall, ...)."""
    if not isinstance(tool_call_obj, dict):
        return "unknown"
    for key, value in tool_call_obj.items():
        if isinstance(value, dict) and key.endswith("ToolCall"):
            return key[: -len("ToolCall")] or key
        if key == "function" and isinstance(value, dict):
            return str(value.get("name") or "function")
    return "unknown"


def _tool_args_from_event(tool_call_obj: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool_call_obj, dict):
        return {}
    for key, value in tool_call_obj.items():
        if isinstance(value, dict) and key.endswith("ToolCall"):
            args = value.get("args")
            return args if isinstance(args, dict) else {}
        if key == "function" and isinstance(value, dict):
            raw = value.get("arguments")
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except Exception:
                    return {"raw": raw}
            if isinstance(raw, dict):
                return raw
    return {}


def _normalize_cursor_tool_name(name: str) -> str:
    """Map cursor tool names onto our canonical trace flags."""
    n = (name or "").lower()
    if n in {"read", "readtool"}:
        return "read_file"
    if n in {"write", "writetool"}:
        return "write_file"
    if n in {"list", "listtool", "ls"}:
        return "list_files"
    if n in {"shell", "terminal", "run", "bash"}:
        return "shell"
    return name or "unknown"


def run_cursor_agent(
    model: str,
    user_prompt: str,
    workspace: str,
    api_key: str | None = None,
    circuit_data: dict[str, Any] | None = None,
    answer_format: str = "letter_only",
    return_trace: bool = False,
    no_tools: bool = False,
    timeout: float = 240.0,
) -> str | dict[str, Any]:
    """Run the cursor-agent CLI on a single MCQ.

    Parameters
    ----------
    no_tools:
        When True, runs in `ask` mode (read-only) and prepends a strong
        instruction discouraging any tool use, so cursor behaves close to a
        plain chat completion. When False, runs in default agent mode with
        full tool access (`--force` auto-approves shell commands).
    """

    binary = _find_cursor_agent_binary()
    if not binary:
        raise RuntimeError(
            "cursor-agent CLI not found on PATH. Install it via "
            "`curl https://cursor.com/install -fsS | bash` and retry."
        )

    key = (api_key or "").strip() or os.environ.get("CURSOR_API_KEY") or ""
    if not key:
        raise RuntimeError("Cursor API key required for cursor-agent runs.")

    if not os.path.isdir(workspace):
        os.makedirs(workspace, exist_ok=True)

    system_prompt = build_full_system_prompt(
        circuit_data=circuit_data,
        answer_format=answer_format,
    )

    if no_tools:
        no_tool_directive = (
            "STRICT NO-TOOL MODE: answer using ONLY your own knowledge. "
            "Do NOT call any tools. Do NOT read or write any files. "
            "Do NOT search the codebase. Do NOT execute any shell commands. "
            "Reply with ONLY the option letter (A, B, C, or D)."
        )
        combined_prompt = f"{system_prompt}\n\n{no_tool_directive}\n\n{user_prompt}"
    else:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    cli_model = _strip_cursor_prefix(model)

    args: list[str] = [
        binary,
        "--print",
        "--output-format", "stream-json",
        "--workspace", workspace,
        "--model", cli_model,
        "--api-key", key,
        "--trust",
    ]
    if no_tools:
        args.extend(["--mode", "ask"])
    else:
        args.append("--force")

    args.append(combined_prompt)

    trace = _build_trace(model)
    if no_tools:
        trace["mode"] = "cursor_no_tool"
    else:
        trace["mode"] = "cursor_agent"

    try:
        proc = subprocess.run(
            args,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        trace["errors"].append(f"cursor-agent timed out after {timeout:.0f}s")
        msg = f"ERROR: cursor-agent timed out: {exc}"
        if return_trace:
            return {"answer": msg, "trace": trace}
        return msg

    final_text = ""
    last_assistant_text = ""
    pending_tools: dict[str, dict[str, Any]] = {}

    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        if etype == "assistant":
            try:
                content = event["message"]["content"]
                if isinstance(content, list):
                    for piece in content:
                        if isinstance(piece, dict) and piece.get("type") == "text":
                            last_assistant_text = piece.get("text", "") or last_assistant_text
            except Exception:
                pass
        elif etype == "tool_call":
            sub = event.get("subtype")
            call_id = event.get("call_id") or ""
            tool_obj = event.get("tool_call") or {}
            raw_name = _tool_name_from_event(tool_obj)
            tool_args = _tool_args_from_event(tool_obj)
            canonical = _normalize_cursor_tool_name(raw_name)

            if sub == "started":
                pending_tools[call_id] = {
                    "name": canonical,
                    "args": tool_args,
                }
            elif sub == "completed":
                started = pending_tools.pop(call_id, {"name": canonical, "args": tool_args})
                trace["used_tools"] = True
                trace["tool_call_count"] += 1
                if canonical == "write_file":
                    trace["wrote_files"] = True
                elif canonical == "read_file":
                    trace["read_files"] = True
                elif canonical == "list_files":
                    trace["listed_files"] = True
                elif canonical == "shell":
                    trace["simulation_executed"] = True

                # Try to surface success/error from the result envelope.
                result_envelope: dict[str, Any] = {}
                inner = tool_obj
                for key2, value in (inner or {}).items():
                    if isinstance(value, dict) and "result" in value:
                        result_envelope = value.get("result") or {}
                        break
                success = bool(result_envelope.get("success"))

                trace["tool_calls"].append({
                    "name": started["name"],
                    "args": started["args"],
                    "success": success,
                    "content_preview": "",
                })
                if not success and result_envelope:
                    trace["errors"].append(f"{started['name']} failed")
        elif etype == "result":
            final_text = event.get("result") or final_text
        elif etype == "system" and event.get("subtype") == "init":
            trace["session_id"] = event.get("session_id")
            trace["cursor_model"] = event.get("model")

    if not final_text:
        final_text = last_assistant_text

    if proc.returncode != 0 and not final_text:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-5:]
        err_msg = "cursor-agent exited with code " + str(proc.returncode)
        if stderr_tail:
            err_msg += ": " + " | ".join(stderr_tail)
        trace["errors"].append(err_msg)
        final_text = f"ERROR: {err_msg}"

    answer = (final_text or "").strip()
    if return_trace:
        return {"answer": answer, "trace": trace}
    return answer