"""
VeraGrid Agent System Prompt Builder
=====================================
Generates the agent's system prompt dynamically from:
  1. A static "tool behavior" section (like PHREEQC's)
  2. A dynamic circuit context section (from the frontend)

The circuit data is NOT hardcoded — it's injected at runtime
from whatever the user has built in the frontend.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


# ---------------------------------------------------------------------------
# STATIC PART: Agent identity, tool descriptions, behavioral rules
# This never changes — it's the agent's "operating manual"
# ---------------------------------------------------------------------------

STATIC_SYSTEM_PROMPT = """\
You are a VeraGrid power-flow agent.
All actions must occur within the allowed workspace folder.

Workspace rules:
- You may do file operations under the allowed workspace folder with all \
the tools (use relative paths).
- Do not assume any input files exist; create any needed input files yourself.
- Never reference or attempt to access files outside the workspace.

Enabled tools:
- list_files(path): List files/directories under a workspace directory. \
Use this first when paths are unclear.
- read_file(path, start_line?, end_line?): Read a text file in the workspace. \
Small files are returned in full. Large files are auto-truncated to head/tail \
preview with total line count. Use start_line and end_line (1-based) to read \
specific sections of large files.
- write_file(filename, content): Create or overwrite a workspace text file. \
Use full file content each time.
- run_web_opf(input_file?, results_file?, vnom?): Run AC-OPF through \
VeraGridEngine via web_opf_agent.py and return converged flag, iterations, \
summary metrics, and a JSON results file path. Use this as the PRIMARY \
simulation tool for MCQ evaluation questions.
- execute_veragrid(input_file, solver?): Run the VeraGrid power-flow solver \
on the given input file. Input path must exist in workspace. Returns returncode, \
stdout/stderr tail, output file paths/sizes, and a table-of-contents (section \
header + line number) for the result file. IMPORTANT: File contents are NOT \
included. Use read_file with start_line/end_line to jump directly to the \
section you need (e.g. "Bus Voltages", "Branch Flows", "Losses Summary").

Solver options for execute_veragrid:
- NR (Newton-Raphson, default — most accurate)
- FDXB / FDBX (Fast-Decoupled — faster, slightly less accurate)
- DC (DC approximation — fastest, no reactive power)
- GS (Gauss-Seidel — educational, slow convergence)

Tool-first behavior (REQUIRED):
- For ANY quantitative or numerical power-flow question, you MUST use tools \
to compute the answer. Do NOT answer numerical questions from memory.
- Preferred workflow for MCQ evaluation: write input model JSON with \
write_file → run_web_opf → read_file on returned OPF JSON → extract answer.
- Alternate workflow: write a VeraGrid input file with write_file → \
execute_veragrid → use read_file to inspect relevant output sections.
- If a tool fails, inspect the error, fix the input, and retry. Common \
errors: missing slack bus, disconnected buses, bad impedance values.
- Call only one tool per turn.
- Prefer using tools rather than asking the user to provide data.
- After execution, state the key result, then give your final answer.

VeraGrid input file format (JSON):
{
  "name": "case_name",
  "baseMVA": 100,
  "buses": [
    {
      "id": <int>,
      "type": "Slack" | "PV" | "PQ",
      "Vm": <float, voltage magnitude setpoint in pu>,
      "Va": <float, voltage angle in degrees>,
      "Pd": <float, active load in MW>,
      "Qd": <float, reactive load in MVAr>,
      "Pg": <float, active generation in MW (PV/Slack buses)>,
      "Qg": <float, reactive generation in MVAr>
    }
  ],
  "branches": [
    {
      "from": <int, from bus id>,
      "to": <int, to bus id>,
      "r": <float, resistance in pu>,
      "x": <float, reactance in pu>,
      "b": <float, total line charging susceptance in pu>,
      "rateA": <float, MW rating, optional>
    }
  ],
  "generators": [
    {
      "bus": <int, bus id>,
      "Pg": <float, MW>,
      "Qg": <float, MVAr>,
      "Vg": <float, voltage setpoint pu>,
      "Pmax": <float, optional>,
      "Pmin": <float, optional>,
      "Qmax": <float, optional>,
      "Qmin": <float, optional>
    }
  ]
}

Important conventions:
- Bus type "Slack": voltage magnitude and angle are fixed (reference bus). \
Every system must have exactly one Slack bus.
- Bus type "PV": active power and voltage magnitude are specified.
- Bus type "PQ": active and reactive power (load) are specified.
- Impedances (r, x, b) are in per-unit on the system baseMVA.
- If the question describes a modification to the circuit (e.g., removing \
a line, changing a load), modify the input file accordingly before running.

If a file does not exist, explain what files are available using list_files.
"""


# ---------------------------------------------------------------------------
# DYNAMIC PART: Circuit context injected from the frontend
# ---------------------------------------------------------------------------

def build_circuit_context(circuit_data: dict[str, Any]) -> str:
    """
    Convert frontend circuit data into a natural-language context block
    that gets appended to the system prompt.

    This is called at runtime — every time the user asks a question,
    the frontend sends the current circuit state, and this function
    translates it into text the agent can understand.

    Parameters
    ----------
    circuit_data : dict
        The circuit as represented in the frontend. Expected structure:
        {
            "name": "3-bus radial",
            "baseMVA": 100,
            "buses": [...],
            "branches": [...],
            "generators": [...]
        }
    """
    if not circuit_data:
        return (
            "\nNo circuit is currently loaded in the frontend. "
            "If the question describes a circuit, build the input file from "
            "the question description. If not, ask the user to provide or "
            "load a circuit.\n"
        )

    lines = []
    lines.append("\n--- ACTIVE CIRCUIT (from frontend) ---")
    name = circuit_data.get("name", "unnamed")
    base_mva = circuit_data.get("baseMVA", 100)
    lines.append(f"Circuit name: {name}")
    lines.append(f"System base: {base_mva} MVA")

    # Buses
    buses = circuit_data.get("buses", [])
    lines.append(f"\nBuses ({len(buses)} total):")
    for bus in buses:
        bid = bus.get("id", "?")
        btype = bus.get("type", "PQ")
        vm = bus.get("Vm", 1.0)
        pd = bus.get("Pd", 0)
        qd = bus.get("Qd", 0)
        pg = bus.get("Pg", 0)

        desc = f"  Bus {bid} ({btype})"
        if btype == "Slack":
            desc += f": V = {vm} pu (reference)"
        elif btype == "PV":
            desc += f": Pg = {pg} MW, V = {vm} pu"
        else:  # PQ
            desc += f": Pd = {pd} MW, Qd = {qd} MVAr"

        if pd > 0 and btype != "PQ":
            desc += f", Load: {pd} MW + j{qd} MVAr"
        lines.append(desc)

    # Branches
    branches = circuit_data.get("branches", [])
    lines.append(f"\nBranches ({len(branches)} total):")
    for br in branches:
        f_bus = br.get("from", "?")
        t_bus = br.get("to", "?")
        r = br.get("r", 0)
        x = br.get("x", 0)
        b = br.get("b", 0)
        lines.append(f"  Line {f_bus}-{t_bus}: r={r}, x={x}, b={b} (pu)")

    # Generators
    generators = circuit_data.get("generators", [])
    if generators:
        lines.append(f"\nGenerators ({len(generators)} total):")
        for gen in generators:
            gbus = gen.get("bus", "?")
            pg = gen.get("Pg", 0)
            vg = gen.get("Vg", 1.0)
            lines.append(f"  Gen at Bus {gbus}: Pg={pg} MW, Vg={vg} pu")

    lines.append("\n--- END ACTIVE CIRCUIT ---")

    lines.append(
        "\nWhen answering questions about this circuit, use the data above "
        "to build your VeraGrid input file. The circuit data above is the "
        "ground truth — use it exactly as specified, do not modify unless "
        "the question explicitly asks for a modification (e.g., 'if Line "
        "2-3 is removed...')."
    )
    lines.append(
        "\nAn active circuit file is available in the workspace as "
        "'active_circuit_model.json'. You may run run_web_opf on it directly."
    )

    return "\n".join(lines)


def build_full_system_prompt(
    circuit_data: dict[str, Any] | None = None,
    answer_format: str = "letter_only",
) -> str:
    """
    Build the complete system prompt by combining:
    1. Static agent instructions (tools, rules, format)
    2. Dynamic circuit context (from frontend)
    3. Answer format instructions

    Parameters
    ----------
    circuit_data : dict or None
        Current circuit from the frontend. If None, the agent is told
        no circuit is loaded.
    answer_format : str
        "letter_only" — respond with just A/B/C/D (for MCQ evaluation)
        "explained"   — show work and explain reasoning (for interactive use)
    """
    parts = [STATIC_SYSTEM_PROMPT]

    # Dynamic circuit context
    parts.append(build_circuit_context(circuit_data or {}))

    # Answer format
    if answer_format == "letter_only":
        parts.append(
            "\nAnswer format: After computing the result with tools, respond "
            "with ONLY the option letter (A, B, C, or D). No explanation."
        )
    elif answer_format == "explained":
        parts.append(
            "\nAnswer format: Show your work. Explain which tool calls you "
            "made and why, what the results showed, and how you arrived at "
            "your answer. Then state your final answer clearly."
        )

    return "\n".join(parts)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v


def frontend_model_to_circuit_data(frontend_model: dict[str, Any]) -> dict[str, Any]:
    """
    Convert frontend model format:
      { GENERATORS: [...], LOADS: [...], LINES: [...] }
    into prompt-builder circuit format:
      { name, baseMVA, buses, branches, generators }.
    """
    model = frontend_model or {}
    gens = model.get("GENERATORS") or []
    loads = model.get("LOADS") or []
    lines = model.get("LINES") or []

    bus_ids: set[int] = set()
    for g in gens:
        bus_ids.add(int(_num(g.get("bus"), 0)))
    for ld in loads:
        bus_ids.add(int(_num(ld.get("bus"), 0)))
    for ln in lines:
        bus_ids.add(int(_num(ln.get("from"), 0)))
        bus_ids.add(int(_num(ln.get("to"), 0)))

    buses: list[dict[str, Any]] = []
    for bus_id in sorted(bus_ids):
        gen = next((g for g in gens if int(_num(g.get("bus"), 0)) == bus_id), None)
        load = next((l for l in loads if int(_num(l.get("bus"), 0)) == bus_id), None)

        is_slack = bool(gen and "slack" in str(gen.get("name", "")).lower())
        btype = "Slack" if is_slack else ("PV" if gen else "PQ")
        vm = _num((gen or {}).get("vset"), 1.0)
        pd = _num((load or {}).get("P"), 0.0)
        qd = _num((load or {}).get("Q"), 0.0)

        if gen:
            pmin = _num(gen.get("Pmin"), 0.0)
            pmax = _num(gen.get("Pmax"), max(0.0, pmin))
            pg = (pmin + pmax) / 2.0
        else:
            pg = 0.0

        buses.append({
            "id": bus_id,
            "type": btype,
            "Vm": vm,
            "Va": 0.0,
            "Pd": pd,
            "Qd": qd,
            "Pg": pg,
            "Qg": 0.0,
        })

    branches = []
    for ln in lines:
        branches.append({
            "from": int(_num(ln.get("from"), 0)),
            "to": int(_num(ln.get("to"), 0)),
            "r": _num(ln.get("r"), 0.0),
            "x": _num(ln.get("x"), 0.0),
            "b": _num(ln.get("b"), 0.0),
        })

    generators = []
    for g in gens:
        pmin = _num(g.get("Pmin"), 0.0)
        pmax = _num(g.get("Pmax"), max(0.0, pmin))
        generators.append({
            "bus": int(_num(g.get("bus"), 0)),
            "Pg": (pmin + pmax) / 2.0,
            "Qg": 0.0,
            "Vg": _num(g.get("vset"), 1.0),
            "Pmax": pmax,
            "Pmin": pmin,
        })

    return {
        "name": "frontend-circuit",
        "baseMVA": 100,
        "buses": buses,
        "branches": branches,
        "generators": generators,
    }


# ---------------------------------------------------------------------------
# Example: how the frontend would call this
# ---------------------------------------------------------------------------

def example_usage():
    """
    Demonstrates how the frontend sends circuit data and gets a prompt.
    In your real app, this happens via an API call from the frontend.
    """

    # This is what your frontend sends — the current circuit state
    frontend_circuit = {
        "name": "3-bus radial",
        "baseMVA": 100,
        "buses": [
            {"id": 1, "type": "Slack", "Vm": 1.0, "Va": 0, "Pd": 0, "Qd": 0},
            {"id": 2, "type": "PQ", "Vm": 1.0, "Va": 0, "Pd": 40, "Qd": 20},
            {"id": 3, "type": "PQ", "Vm": 1.0, "Va": 0, "Pd": 25, "Qd": 15},
        ],
        "branches": [
            {"from": 1, "to": 2, "r": 0.05, "x": 0.11, "b": 0.02},
            {"from": 2, "to": 3, "r": 0.04, "x": 0.09, "b": 0.02},
        ],
        "generators": [
            {"bus": 1, "Pg": 0, "Qg": 0, "Vg": 1.0},
        ],
    }

    # Build the system prompt dynamically
    prompt = build_full_system_prompt(
        circuit_data=frontend_circuit,
        answer_format="letter_only",
    )

    print("=" * 70)
    print("GENERATED SYSTEM PROMPT")
    print("=" * 70)
    print(prompt)
    print("=" * 70)
    print(f"\nPrompt length: {len(prompt)} chars")

    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VeraGrid system prompt from circuit data")
    parser.add_argument(
        "--frontend-model-file",
        default="",
        help="Path to frontend model JSON with GENERATORS/LOADS/LINES",
    )
    parser.add_argument(
        "--circuit-file",
        default="",
        help="Path to circuit JSON already in {buses,branches,generators} format",
    )
    parser.add_argument(
        "--answer-format",
        choices=["letter_only", "explained"],
        default="letter_only",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path for generated prompt",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print prompt only (legacy behavior).",
    )
    args = parser.parse_args()

    if not args.frontend_model_file and not args.circuit_file:
        if args.print_only:
            example_usage()
        else:
            prompt = build_full_system_prompt(circuit_data=None, answer_format=args.answer_format)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(prompt)
            print(prompt)
    else:
        if args.frontend_model_file:
            with open(args.frontend_model_file, "r", encoding="utf-8") as f:
                frontend_model = json.load(f)
            circuit_data = frontend_model_to_circuit_data(frontend_model)
        else:
            with open(args.circuit_file, "r", encoding="utf-8") as f:
                circuit_data = json.load(f)

        prompt = build_full_system_prompt(circuit_data=circuit_data, answer_format=args.answer_format)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(prompt)
        print(prompt)