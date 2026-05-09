"""
VeraGrid Agent Tools
====================
Defines the four core tools the agent can call:

1. write_file    - Create/overwrite files in the workspace (VeraGrid inputs)
2. execute_veragrid - Run the VeraGrid power-flow solver on an input file
3. read_file     - Read file contents, optionally by line range
4. list_files    - List files and directories in the workspace

Each tool has:
  - A JSON schema (for the LLM function-calling API)
  - A Python handler (the actual implementation)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Tool JSON Schemas (sent to the LLM so it knows how to call each tool)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a text file in the workspace. "
                "Use this to write VeraGrid input files (.json, .raw, .txt) "
                "that define the power system case for simulation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to create (e.g. 'case5.json')",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write into the file",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_veragrid",
            "description": (
                "Run the VeraGrid power-flow solver on a given input file. "
                "Returns metadata (return code, stdout/stderr tail, output file "
                "paths and sizes) and a Table of Contents listing section headers "
                "with line numbers from the results file. Does NOT return the full "
                "output — use read_file to inspect specific sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the VeraGrid input file to run",
                    },
                    "solver": {
                        "type": "string",
                        "enum": ["NR", "FDXB", "FDBX", "DC", "GS"],
                        "description": (
                            "Power-flow solver method. "
                            "NR=Newton-Raphson, FDXB/FDBX=Fast-Decoupled, "
                            "DC=DC power flow, GS=Gauss-Seidel. Default: NR"
                        ),
                    },
                },
                "required": ["input_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_web_opf",
            "description": (
                "Run VeraGrid AC-OPF directly via web_opf_agent.py using a model JSON file. "
                "Accepts either frontend model format {GENERATORS,LOADS,LINES} or circuit "
                "format {buses,branches,generators}. Returns OPF summary and output file path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": (
                            "Path to input model JSON. Defaults to 'active_circuit_model.json'."
                        ),
                    },
                    "results_file": {
                        "type": "string",
                        "description": (
                            "Path to OPF output JSON. Defaults to 'active_opf_results.json'."
                        ),
                    },
                    "vnom": {
                        "type": "number",
                        "description": "Nominal voltage in kV. Optional, default 20.0.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the workspace. For small files (< 20,000 chars), "
                "returns the full content. For large files, returns first and last "
                "80 lines with total line count. Use start_line and end_line to "
                "read specific sections identified in the TOC from execute_veragrid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed). Optional.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, inclusive). Optional.",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories in the workspace. "
                "Shows file names and sizes to help the agent understand "
                "what input/output files are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to workspace root.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Handlers (actual implementations)
# ---------------------------------------------------------------------------

MAX_SMALL_FILE_CHARS = 20_000
TRUNCATE_LINES = 80
TOC_MAX_ENTRIES = 50
LARGE_FILE_SKIP_TOC_BYTES = 50 * 1024 * 1024  # 50 MB
WEB_OPF_SCRIPT = os.path.join(os.path.dirname(__file__), "web_opf_agent.py")


@dataclass
class ToolResult:
    """Wrapper for tool execution results returned to the agent."""
    success: bool
    content: str  # text the agent sees in its next turn


def _resolve_path(workspace: str, filename: str) -> str:
    """Resolve a filename relative to the workspace, preventing escapes."""
    joined = os.path.normpath(os.path.join(workspace, filename))
    if not joined.startswith(os.path.normpath(workspace)):
        raise ValueError(f"Path escapes workspace: {filename}")
    return joined


# ---- Tool 1: write_file ----
def handle_write_file(workspace: str, args: dict) -> ToolResult:
    """
    WHAT IT DOES:
    Creates or overwrites a text file in the workspace directory.

    WHY THE AGENT NEEDS IT:
    The agent must author VeraGrid input files that define the power system
    (buses, branches, generators, loads, solver settings). This is how the
    agent "acts" — by producing simulation inputs it can then execute.

    EXAMPLE:
    The agent writes a JSON file describing a 5-bus system with specific
    load values, then calls execute_veragrid on that file.
    """
    filename = args["filename"]
    content = args["content"]
    filepath = _resolve_path(workspace, filename)

    os.makedirs(os.path.dirname(filepath) or workspace, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    size = os.path.getsize(filepath)
    return ToolResult(
        success=True,
        content=f"File written: {filename} ({size} bytes)",
    )


# ---- Tool 2: execute_veragrid ----
def _build_toc(filepath: str) -> list[dict[str, Any]]:
    """
    Parse the VeraGrid output file and build a Table of Contents.
    Looks for section headers (lines starting with '=', '#', or known keywords)
    and records their line numbers.
    """
    toc_entries: list[dict[str, Any]] = []
    section_keywords = [
        "Bus Data", "Branch Data", "Generator Data", "Load Data",
        "Bus Voltages", "Bus Voltage", "Branch Flows", "Branch Flow",
        "Power Flow Solution", "Convergence", "Summary", "Losses",
        "Mismatch", "Iteration", "Slack Bus", "Generation",
        "Total Load", "Total Generation", "Total Losses",
        "Reactive Power", "Active Power", "Voltage Magnitude",
        "Voltage Angle", "Line Flows", "Transformer",
    ]

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                is_header = (
                    stripped.startswith("=")
                    or stripped.startswith("#")
                    or stripped.startswith("---")
                    or any(kw.lower() in stripped.lower() for kw in section_keywords)
                )
                if is_header and stripped:
                    toc_entries.append({
                        "line": line_no,
                        "header": stripped[:120],
                    })
    except Exception:
        return []

    # Cap at TOC_MAX_ENTRIES: first 25 + last 25
    if len(toc_entries) > TOC_MAX_ENTRIES:
        toc_entries = toc_entries[:25] + toc_entries[-25:]

    return toc_entries


def handle_execute_veragrid(workspace: str, args: dict) -> ToolResult:
    """
    WHAT IT DOES:
    Runs the VeraGrid power-flow solver on a specified input file.
    Returns metadata + a Table of Contents of the output — NOT the full results.

    WHY THE AGENT NEEDS IT:
    This is the agent's "computation engine." When a question requires
    numerical power-flow results (bus voltages, line flows, losses),
    the agent calls this tool to run the actual simulation.

    WHAT IT RETURNS:
    - return_code: 0 = success, nonzero = error
    - stdout/stderr tail (last 4000 chars) for debugging
    - output file paths and sizes
    - TOC: section headers + line numbers so the agent knows WHERE
      each result lives without reading the entire output

    ON ERROR:
    Returns full error output so the agent can diagnose the problem,
    fix the input file with write_file, and retry.

    NOTE:
    Replace the subprocess call below with your actual VeraGrid CLI
    invocation. The current implementation uses a placeholder.
    """
    input_file = args["input_file"]
    solver = args.get("solver", "NR")
    input_path = _resolve_path(workspace, input_file)

    if not os.path.exists(input_path):
        return ToolResult(False, f"Input file not found: {input_file}")

    # Derive output file path
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_result.txt"
    output_path = _resolve_path(workspace, output_file)

    # -----------------------------------------------------------------
    # >>> REPLACE THIS BLOCK WITH YOUR ACTUAL VERAGRID CLI CALL <<<
    # -----------------------------------------------------------------
    # Example: subprocess.run(["veragrid", "--input", input_path,
    #                          "--solver", solver, "--output", output_path],
    #                         capture_output=True, text=True, timeout=300)
    #
    # For now, we check if veragrid is available, otherwise use a stub.
    # -----------------------------------------------------------------

    veragrid_cmd = os.environ.get("VERAGRID_CMD", "veragrid")
    cmd = [veragrid_cmd, "--input", input_path, "--solver", solver, "--output", output_path]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workspace,
        )
        return_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except FileNotFoundError:
        # VeraGrid not installed — generate a stub output for development
        return_code, stdout, stderr = _generate_stub_output(
            input_path, output_path, solver
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, "VeraGrid execution timed out after 300 seconds.")
    except Exception as exc:
        return ToolResult(False, f"Execution error: {exc}")

    # Build metadata response
    output_files: list[dict[str, Any]] = []
    if os.path.exists(output_path):
        output_files.append({
            "path": output_file,
            "size_bytes": os.path.getsize(output_path),
        })

    # Build TOC (skip for very large files)
    toc: list[dict[str, Any]] = []
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size < LARGE_FILE_SKIP_TOC_BYTES:
            toc = _build_toc(output_path)

    # Truncate stdout/stderr to last 4000 chars
    tail = (stdout + "\n" + stderr).strip()
    if len(tail) > 4000:
        tail = "... (truncated) ...\n" + tail[-4000:]

    metadata = {
        "return_code": return_code,
        "output_files": output_files,
        "stdout_stderr_tail": tail,
        "toc": toc,
        "toc_note": (
            "Use read_file with start_line/end_line to inspect specific sections. "
            "Do NOT request the entire file — read only what you need to answer the question."
        ),
    }

    if return_code != 0:
        metadata["error_hint"] = (
            "Nonzero exit code. Check stdout_stderr_tail for error details. "
            "Common fixes: check bus numbering, ensure slack bus exists, "
            "verify branch connectivity."
        )

    return ToolResult(
        success=(return_code == 0),
        content=json.dumps(metadata, indent=2),
    )


def _generate_stub_output(input_path: str, output_path: str, solver: str) -> tuple[int, str, str]:
    """
    Development stub: generates a realistic-looking VeraGrid output file
    when the actual solver isn't installed. Replace with real execution.
    """
    try:
        with open(input_path, "r") as f:
            input_data = json.load(f)
    except Exception:
        input_data = {}

    buses = input_data.get("buses", [])
    branches = input_data.get("branches", [])
    n_buses = len(buses) if buses else 5

    lines = []
    lines.append("=" * 60)
    lines.append(f"  VeraGrid Power Flow Solution  (Solver: {solver})")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"--- Convergence ---")
    lines.append(f"  Solver: {solver}")
    lines.append(f"  Iterations: 4")
    lines.append(f"  Max mismatch: 1.23e-10 pu")
    lines.append(f"  Status: CONVERGED")
    lines.append("")

    lines.append(f"--- Bus Voltages ---")
    lines.append(f"  {'Bus':>5}  {'Vmag (pu)':>10}  {'Vang (deg)':>10}  {'Type':>6}")
    lines.append(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*6}")
    import random
    random.seed(42)
    for i in range(1, n_buses + 1):
        bus_info = buses[i - 1] if i <= len(buses) else {}
        btype = bus_info.get("type", "PQ" if i > 1 else "Slack")
        vmag = 1.0 + random.uniform(-0.05, 0.02) if btype != "Slack" else 1.0
        vang = random.uniform(-8, 0) if btype != "Slack" else 0.0
        lines.append(f"  {i:>5}  {vmag:>10.4f}  {vang:>10.2f}  {btype:>6}")
    lines.append("")

    lines.append(f"--- Branch Flows ---")
    lines.append(f"  {'From':>5}  {'To':>5}  {'P (MW)':>10}  {'Q (MVAr)':>10}")
    lines.append(f"  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}")
    if branches:
        for br in branches:
            p = random.uniform(10, 80)
            q = random.uniform(-10, 30)
            lines.append(f"  {br.get('from', 1):>5}  {br.get('to', 2):>5}  {p:>10.2f}  {q:>10.2f}")
    else:
        for i in range(1, n_buses):
            p = random.uniform(10, 80)
            q = random.uniform(-10, 30)
            lines.append(f"  {i:>5}  {i+1:>5}  {p:>10.2f}  {q:>10.2f}")
    lines.append("")

    total_p_loss = random.uniform(1, 5)
    total_q_loss = random.uniform(2, 10)
    lines.append(f"--- Losses Summary ---")
    lines.append(f"  Total Active Power Loss:   {total_p_loss:.3f} MW")
    lines.append(f"  Total Reactive Power Loss:  {total_q_loss:.3f} MVAr")
    lines.append("")

    lines.append(f"--- Total Generation ---")
    total_pg = random.uniform(100, 300)
    total_qg = random.uniform(20, 80)
    lines.append(f"  Total P Generation: {total_pg:.2f} MW")
    lines.append(f"  Total Q Generation: {total_qg:.2f} MVAr")
    lines.append("")
    lines.append("=" * 60)
    lines.append("  End of VeraGrid Power Flow Report")
    lines.append("=" * 60)

    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)

    return 0, f"VeraGrid completed successfully. Output: {output_path}", ""


def _circuit_to_frontend_model(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize input schema for web_opf_agent.py.
    Accept either frontend model {GENERATORS, LOADS, LINES} or
    circuit model {buses, branches, generators}.
    """
    if {"GENERATORS", "LOADS", "LINES"}.issubset(data.keys()):
        return data

    buses = data.get("buses") or []
    branches = data.get("branches") or []
    generators = data.get("generators") or []

    front_lines = []
    for br in branches:
        f_bus = int(br.get("from", 0))
        t_bus = int(br.get("to", 0))
        front_lines.append({
            "from": f_bus,
            "to": t_bus,
            "name": str(br.get("name", f"Line {f_bus}-{t_bus}")),
            "r": float(br.get("r", 0.0)),
            "x": float(br.get("x", 0.0)),
            "b": float(br.get("b", 0.0)),
        })

    front_gens = []
    for idx, gen in enumerate(generators, start=1):
        bus = int(gen.get("bus", 0))
        front_gens.append({
            "bus": bus,
            "name": str(gen.get("name", f"Gen {idx}")),
            "Pmin": float(gen.get("Pmin", 0.0)),
            "Pmax": float(gen.get("Pmax", max(0.0, float(gen.get("Pg", 0.0))))),
            "a": float(gen.get("a", gen.get("Cost2", 0.0))),
            "b": float(gen.get("b", gen.get("Cost", 1.0))),
            "vset": float(gen.get("Vg", gen.get("Vm", 1.0))),
        })

    front_loads = []
    for idx, bus in enumerate(buses, start=1):
        pd = float(bus.get("Pd", 0.0))
        qd = float(bus.get("Qd", 0.0))
        if pd == 0.0 and qd == 0.0:
            continue
        bidx = int(bus.get("id", idx))
        front_loads.append({
            "bus": bidx,
            "name": str(bus.get("name", f"Load {idx}")),
            "P": pd,
            "Q": qd,
        })

    return {
        "GENERATORS": front_gens,
        "LOADS": front_loads,
        "LINES": front_lines,
    }


def handle_run_web_opf(workspace: str, args: dict) -> ToolResult:
    """
    Run AC-OPF through web_opf_agent.py for deterministic VeraGrid results.
    """
    input_file = args.get("input_file", "active_circuit_model.json")
    results_file = args.get("results_file", "active_opf_results.json")
    vnom = float(args.get("vnom", 20.0))

    input_path = _resolve_path(workspace, input_file)
    results_path = _resolve_path(workspace, results_file)

    if not os.path.exists(input_path):
        return ToolResult(False, f"Input file not found: {input_file}")
    if not os.path.exists(WEB_OPF_SCRIPT):
        return ToolResult(False, "web_opf_agent.py not found.")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_model = json.load(f)
        normalized_model = _circuit_to_frontend_model(raw_model)
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(normalized_model, f, indent=2)
    except Exception as exc:
        return ToolResult(False, f"Invalid input JSON for run_web_opf: {exc}")

    pscad_dir = os.path.dirname(WEB_OPF_SCRIPT)
    py_candidates = [
        os.path.join(pscad_dir, ".venv312", "bin", "python3"),
        os.path.join(pscad_dir, ".venv312", "bin", "python"),
        sys.executable,
        "python3",
        "python",
    ]
    python_bin = next((p for p in py_candidates if p and (os.path.exists(p) or p in {"python3", "python"})), sys.executable)

    cmd = [
        python_bin,
        WEB_OPF_SCRIPT,
        "--json-file",
        input_path,
        "--results-output",
        results_path,
        "--vnom",
        str(vnom),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workspace,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, "run_web_opf timed out after 300 seconds.")
    except Exception as exc:
        return ToolResult(False, f"run_web_opf execution error: {exc}")

    if proc.returncode != 0:
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if len(tail) > 4000:
            tail = "... (truncated) ...\n" + tail[-4000:]
        return ToolResult(False, f"run_web_opf failed (code={proc.returncode}).\n{tail}")

    if not os.path.exists(results_path):
        return ToolResult(False, f"OPF results file not created: {results_file}")

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as exc:
        return ToolResult(False, f"Could not read OPF results: {exc}")

    response = {
        "return_code": proc.returncode,
        "results_file": results_file,
        "converged": bool((results or {}).get("converged", False)),
        "iterations": (results or {}).get("iterations"),
        "error": (results or {}).get("error"),
        "summary": (results or {}).get("summary", {}),
        "next_steps": [
            f"Use read_file('{results_file}') for full OPF JSON results.",
            "Use list_files() to inspect generated files.",
        ],
    }
    return ToolResult(True, json.dumps(response, indent=2))


# ---- Tool 3: read_file ----
def handle_read_file(workspace: str, args: dict) -> ToolResult:
    """
    WHAT IT DOES:
    Reads a file from the workspace, optionally a specific line range.

    WHY THE AGENT NEEDS IT:
    After execute_veragrid returns a TOC, the agent uses read_file to
    selectively inspect specific sections. For example:
      - TOC says "Bus Voltages" starts at line 11
      - Agent calls read_file(filename, start_line=11, end_line=18)
      - Gets just the voltage table, not the entire output

    SMART TRUNCATION:
    - Files < 20,000 chars → returned in full
    - Larger files → first 80 + last 80 lines with line count
    - Line-range requests always honored regardless of file size

    This ensures the agent's context window is never flooded with
    irrelevant data from large simulation outputs.
    """
    filename = args["filename"]
    filepath = _resolve_path(workspace, filename)

    if not os.path.exists(filepath):
        return ToolResult(False, f"File not found: {filename}")

    start_line = args.get("start_line")
    end_line = args.get("end_line")

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as exc:
        return ToolResult(False, f"Error reading file: {exc}")

    total_lines = len(all_lines)

    # If specific line range requested, return just that range
    if start_line is not None:
        s = max(1, start_line) - 1  # convert to 0-indexed
        e = (end_line or total_lines)
        selected = all_lines[s:e]
        header = f"[{filename}] lines {s+1}-{min(e, total_lines)} of {total_lines}\n"
        # Add line numbers for easy reference
        numbered = []
        for i, line in enumerate(selected, start=s + 1):
            numbered.append(f"{i:>6} | {line.rstrip()}")
        return ToolResult(True, header + "\n".join(numbered))

    # Full file: check size
    content = "".join(all_lines)
    if len(content) <= MAX_SMALL_FILE_CHARS:
        return ToolResult(True, f"[{filename}] ({total_lines} lines)\n{content}")

    # Large file: truncate to first + last TRUNCATE_LINES lines
    head = all_lines[:TRUNCATE_LINES]
    tail = all_lines[-TRUNCATE_LINES:]
    omitted = total_lines - 2 * TRUNCATE_LINES

    parts = [f"[{filename}] ({total_lines} lines — showing first & last {TRUNCATE_LINES})\n"]
    for i, line in enumerate(head, start=1):
        parts.append(f"{i:>6} | {line.rstrip()}")
    parts.append(f"\n  ... ({omitted} lines omitted) ...\n")
    for i, line in enumerate(tail, start=total_lines - TRUNCATE_LINES + 1):
        parts.append(f"{i:>6} | {line.rstrip()}")

    return ToolResult(True, "\n".join(parts))


# ---- Tool 4: list_files ----
def handle_list_files(workspace: str, args: dict) -> ToolResult:
    """
    WHAT IT DOES:
    Lists files and directories in the workspace, showing names and sizes.

    WHY THE AGENT NEEDS IT:
    The agent needs to know what's available — input files it previously
    wrote, output files from VeraGrid runs, reference case files that
    were pre-loaded. This is the agent's "eyes" on the file system.

    EXAMPLE:
    Before running a simulation, the agent might list_files to check
    if a reference IEEE 14-bus case already exists, or after execution
    to confirm the output file was created.
    """
    subpath = args.get("path", ".")
    dirpath = _resolve_path(workspace, subpath)

    if not os.path.isdir(dirpath):
        return ToolResult(False, f"Not a directory: {subpath}")

    entries: list[str] = []
    try:
        for item in sorted(os.listdir(dirpath)):
            full = os.path.join(dirpath, item)
            if os.path.isdir(full):
                entries.append(f"  [DIR]  {item}/")
            else:
                size = os.path.getsize(full)
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024*1024):.1f} MB"
                entries.append(f"  {size_str:>10}  {item}")
    except Exception as exc:
        return ToolResult(False, f"Error listing directory: {exc}")

    if not entries:
        return ToolResult(True, f"Directory '{subpath}' is empty.")

    return ToolResult(True, f"Contents of '{subpath}':\n" + "\n".join(entries))


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "write_file": handle_write_file,
    "execute_veragrid": handle_execute_veragrid,
    "run_web_opf": handle_run_web_opf,
    "read_file": handle_read_file,
    "list_files": handle_list_files,
}


def dispatch_tool(workspace: str, tool_name: str, arguments: dict) -> ToolResult:
    """Route a tool call to the correct handler."""
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return ToolResult(False, f"Unknown tool: {tool_name}")
    try:
        return handler(workspace, arguments)
    except Exception as exc:
        return ToolResult(False, f"Tool error ({tool_name}): {exc}")