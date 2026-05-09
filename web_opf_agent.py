#!/usr/bin/env python3
"""
web_opf_agent_full.py
=====================

Fetch a power-system model from the MCQ Builder (GitHub Pages) site,
run AC OPF via VeraGrid, print complete results, AND export opf_results.json
for the MCQ generator pipeline.

Outputs:
  - Console:          full tabular results (buses, branches, generators, loads, summary)
  - opf_results.json: structured JSON consumed by opf_mcq_generator_150.py

Usage:
  python web_opf_agent_full.py
  python web_opf_agent_full.py --url https://shivanshu-11.github.io/-mcqbuilder.veragrid/
  python web_opf_agent_full.py --json-file my_model.json
  python web_opf_agent_full.py --results-output my_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from typing import Any

import numpy as np
import VeraGridEngine.api as vg


DEFAULT_PAGE_URL = "https://shivanshu-11.github.io/-mcqbuilder.veragrid/"


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

def _fetch_text(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "web_opf_agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _js_object_literal_to_json(text: str) -> str:
    text = re.sub(r"([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', text)
    text = re.sub(r":(\.\d+)(?=[,\}\]])", r":0\1", text)
    return text


def _extract_js_circuit_object(js_source: str) -> str | None:
    marker = "{GENERATORS:["
    start = js_source.find(marker)
    if start < 0:
        return None
    depth = 0
    i = start
    in_str = False
    str_quote: str | None = None
    esc = False
    while i < len(js_source):
        c = js_source[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == str_quote:
                in_str = False
        else:
            if c in ('"', "'", "`"):
                in_str = True
                str_quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return js_source[start : i + 1]
        i += 1
    return None


def fetch_model_from_page(page_url: str) -> dict[str, Any]:
    html = _fetch_text(page_url)
    m = re.search(r'src="([^"]+/assets/index-[^"]+\.js)"', html)
    if not m:
        raise RuntimeError("Could not find Vite bundle script URL in page HTML.")
    bundle_path = m.group(1)
    from urllib.parse import urljoin
    bundle_url = urljoin(page_url, bundle_path) if not bundle_path.startswith("http") else bundle_path
    js = _fetch_text(bundle_url)
    lit = _extract_js_circuit_object(js)
    if lit is None:
        raise RuntimeError(
            "Could not find {GENERATORS:[...]} object in JS bundle. "
            "Host a static JSON (--json-url) if the app layout changes."
        )
    return json.loads(_js_object_literal_to_json(lit))


def load_model_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_model_json_url(url: str) -> dict[str, Any]:
    return json.loads(_fetch_text(url))


# ---------------------------------------------------------------------------
# Build MultiCircuit
# ---------------------------------------------------------------------------

def build_multicircuit(data: dict[str, Any], vnom_kv: float = 20.0) -> vg.MultiCircuit:
    gens = data.get("GENERATORS") or []
    loads = data.get("LOADS") or []
    lines = data.get("LINES") or []

    bus_ids: set[int] = set()
    for g in gens:
        bus_ids.add(int(g["bus"]))
    for ld in loads:
        bus_ids.add(int(ld["bus"]))
    for ln in lines:
        bus_ids.add(int(ln["from"]))
        bus_ids.add(int(ln["to"]))

    if not bus_ids:
        raise ValueError("Model has no buses (empty GENERATORS, LOADS, LINES).")

    nmax = max(bus_ids)
    slack_buses: set[int] = set()
    for g in gens:
        name = str(g.get("name", "")).lower()
        if "slack" in name:
            slack_buses.add(int(g["bus"]))

    grid = vg.MultiCircuit(name="MCQ Builder import", Sbase=100.0)
    buses: dict[int, Any] = {}
    for i in range(nmax + 1):
        is_slack = i in slack_buses
        b = vg.Bus(name=f"Bus {i}", Vnom=vnom_kv, is_slack=is_slack)
        grid.add_bus(b)
        buses[i] = b

    for g in gens:
        bi = int(g["bus"])
        name = str(g.get("name", f"Gen@{bi}"))
        pmin = float(g.get("Pmin", 0.0))
        pmax = float(g.get("Pmax", 9999.0))
        vset = float(g.get("vset", 1.0))
        a = float(g.get("a", 0.0))
        b_cost = float(g.get("b", g.get("Cost", 1.0)))
        p0 = max(pmin, min(0.5 * (pmin + pmax), pmax))
        gen = vg.Generator(
            name=name,
            P=p0,
            Pmin=pmin,
            Pmax=pmax,
            vset=vset,
            is_controlled=True,
            Cost=b_cost,
            Cost2=a,
            Snom=max(pmax, 1.0),
            Qmin=-9999.0,
            Qmax=9999.0,
        )
        grid.add_generator(buses[bi], gen)

    for ld in loads:
        bi = int(ld["bus"])
        name = str(ld.get("name", f"Load@{bi}"))
        load = vg.Load(name=name, P=float(ld["P"]), Q=float(ld.get("Q", 0.0)))
        grid.add_load(buses[bi], load)

    for ln in lines:
        fbus = int(ln["from"])
        tbus = int(ln["to"])
        name = str(ln.get("name", f"Line {fbus}-{tbus}"))
        line = vg.Line(
            buses[fbus],
            buses[tbus],
            name=name,
            r=float(ln["r"]),
            x=float(ln["x"]),
            b=float(ln.get("b", 0.0)),
        )
        grid.add_line(line)

    return grid


# ---------------------------------------------------------------------------
# Extract structured results dict (for JSON export)
# ---------------------------------------------------------------------------

def extract_results_dict(res: Any, grid: vg.MultiCircuit) -> dict[str, Any]:
    """
    Extract all OPF results into a plain Python dict suitable for JSON
    serialization. This is the structured output consumed by
    opf_mcq_generator_150.py.
    """
    sb = float(grid.Sbase)
    nbus = len(grid.buses)
    ngen = len(grid.generators)
    nline = len(grid.lines)

    V = res.voltage
    Vm = np.abs(V)
    Va_deg = np.degrees(np.angle(V))

    # Bus power injections
    Sbus = getattr(res, "Sbus", None)
    if Sbus is None:
        try:
            Ybus = grid.get_Ybus()
            Sbus = V * np.conj(Ybus.dot(V))
        except Exception:
            Sbus = np.zeros(nbus, dtype=complex)

    bus_results = []
    for i in range(nbus):
        bus_results.append({
            "index": i,
            "name": grid.buses[i].name,
            "Vm": round(float(Vm[i]), 6),
            "Va_deg": round(float(Va_deg[i]), 4),
            "P_MW": round(float(Sbus[i].real * sb), 4),
            "Q_Mvar": round(float(Sbus[i].imag * sb), 4),
        })

    # Branch results
    Sf = getattr(res, "Sf", None)
    St = getattr(res, "St", None)
    loading = getattr(res, "loading", None)

    branch_results = []
    for k in range(nline):
        ln = grid.lines[k]
        try:
            fi = grid.buses.index(ln.bus_from)
            ti = grid.buses.index(ln.bus_to)
        except Exception:
            fi = ti = -1

        pf = float(Sf[k].real * sb) if Sf is not None and k < len(Sf) else 0.0
        qf = float(Sf[k].imag * sb) if Sf is not None and k < len(Sf) else 0.0
        pt = float(St[k].real * sb) if St is not None and k < len(St) else 0.0
        qt = float(St[k].imag * sb) if St is not None and k < len(St) else 0.0
        ploss = pf + pt
        qloss = qf + qt

        if loading is not None and k < len(loading):
            ld_pct = abs(float(loading[k])) * 100.0
        else:
            rate = getattr(ln, "rate", 0.0) or getattr(ln, "Snom", 0.0)
            sf_mva = abs(complex(Sf[k])) * sb if Sf is not None and k < len(Sf) else 0.0
            ld_pct = (sf_mva / rate * 100.0) if rate > 0 else 0.0

        branch_results.append({
            "index": k,
            "name": ln.name,
            "from_bus": fi,
            "to_bus": ti,
            "Pf_MW": round(pf, 4),
            "Qf_Mvar": round(qf, 4),
            "Pt_MW": round(pt, 4),
            "Qt_Mvar": round(qt, 4),
            "loading_pct": round(ld_pct, 2),
            "Ploss_MW": round(ploss, 4),
            "Qloss_Mvar": round(qloss, 4),
            "r": float(getattr(ln, "R", getattr(ln, "r", 0.0))),
            "x": float(getattr(ln, "X", getattr(ln, "x", 0.0))),
            "b": float(getattr(ln, "B", getattr(ln, "b", 0.0))),
        })

    # Generator results
    Pg = getattr(res, "Pg", None)
    Qg = getattr(res, "Qg", None)
    Pcost = getattr(res, "Pcost", None)

    gen_results = []
    for k in range(ngen):
        g = grid.generators[k]
        try:
            gbi = grid.buses.index(g.bus)
        except Exception:
            gbi = -1

        pg = float(Pg[k] * sb) if Pg is not None and k < len(Pg) else 0.0
        qg = float(Qg[k] * sb) if Qg is not None and k < len(Qg) else float("nan")
        pc = float(Pcost[k]) if Pcost is not None and k < len(Pcost) else 0.0

        gen_results.append({
            "index": k,
            "name": g.name,
            "bus": gbi,
            "P_MW": round(pg, 4),
            "Q_Mvar": round(qg, 4) if not math.isnan(qg) else None,
            "Pmin": float(g.Pmin),
            "Pmax": float(g.Pmax),
            "Cost": float(g.Cost),
            "Cost2": float(g.Cost2),
            "Pcost": round(pc, 6),
        })

    # Load results
    load_results = []
    for k, ld in enumerate(grid.loads):
        try:
            lbi = grid.buses.index(ld.bus)
        except Exception:
            lbi = -1
        load_results.append({
            "index": k,
            "name": ld.name,
            "bus": lbi,
            "P_MW": float(ld.P),
            "Q_Mvar": float(ld.Q),
        })

    # Summary
    total_pg = sum(g["P_MW"] for g in gen_results)
    total_qg = sum(g["Q_Mvar"] for g in gen_results if g["Q_Mvar"] is not None)
    total_pl = sum(ld["P_MW"] for ld in load_results)
    total_ql = sum(ld["Q_Mvar"] for ld in load_results)
    total_ploss = sum(br["Ploss_MW"] for br in branch_results)
    total_qloss = sum(br["Qloss_Mvar"] for br in branch_results)
    total_cost = sum(g["Pcost"] for g in gen_results)

    return {
        "converged": bool(res.converged),
        "iterations": int(res.iterations),
        "error": float(res.error),
        "Sbase_MVA": sb,
        "buses": bus_results,
        "branches": branch_results,
        "generators": gen_results,
        "loads": load_results,
        "summary": {
            "total_gen_P_MW": round(total_pg, 4),
            "total_gen_Q_Mvar": round(total_qg, 4),
            "total_load_P_MW": round(total_pl, 4),
            "total_load_Q_Mvar": round(total_ql, 4),
            "total_Ploss_MW": round(total_ploss, 4),
            "total_Qloss_Mvar": round(total_qloss, 4),
            "total_cost": round(total_cost, 6),
        },
    }


# ---------------------------------------------------------------------------
# Console printer (uses the extracted dict — no duplication of logic)
# ---------------------------------------------------------------------------

def print_separator(title: str, width: int = 110) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_full_results(rd: dict[str, Any]) -> None:
    """Pretty-print results from the extracted results dict."""

    print(f"\nSolver converged: {rd['converged']}")
    print(f"Iterations:       {rd['iterations']}")
    print(f"Error:            {rd['error']:.6e}")

    # Bus results
    print_separator("BUS RESULTS")
    print(f"{'Bus':>6s}  {'Name':24s}  {'Vm (p.u.)':>10s}  {'Va (deg)':>10s}"
          f"  {'P (MW)':>10s}  {'Q (Mvar)':>10s}")
    print("-" * 110)
    for b in rd["buses"]:
        print(f"{b['index']:6d}  {b['name']:24s}  {b['Vm']:10.6f}  {b['Va_deg']:10.4f}"
              f"  {b['P_MW']:10.4f}  {b['Q_Mvar']:10.4f}")

    # Branch results
    print_separator("BRANCH RESULTS")
    print(f"{'#':>4s}  {'Name':24s}  {'From':>5s}  {'To':>5s}"
          f"  {'Pf(MW)':>10s}  {'Qf(Mvar)':>10s}"
          f"  {'Pt(MW)':>10s}  {'Qt(Mvar)':>10s}"
          f"  {'Load(%)':>8s}  {'Ploss(MW)':>10s}  {'Qloss(Mvar)':>12s}")
    print("-" * 140)
    for br in rd["branches"]:
        print(f"{br['index']:4d}  {br['name']:24s}  {br['from_bus']:5d}  {br['to_bus']:5d}"
              f"  {br['Pf_MW']:10.4f}  {br['Qf_Mvar']:10.4f}"
              f"  {br['Pt_MW']:10.4f}  {br['Qt_Mvar']:10.4f}"
              f"  {br['loading_pct']:8.2f}  {br['Ploss_MW']:10.4f}  {br['Qloss_Mvar']:12.4f}")

    # Generator dispatch
    print_separator("GENERATOR DISPATCH")
    print(f"{'#':>4s}  {'Name':24s}  {'Bus':>5s}"
          f"  {'P(MW)':>10s}  {'Pmin':>8s}  {'Pmax':>8s}"
          f"  {'Q(Mvar)':>10s}  {'Pcost':>10s}")
    print("-" * 110)
    for g in rd["generators"]:
        qg_str = f"{g['Q_Mvar']:10.4f}" if g["Q_Mvar"] is not None else "       nan"
        print(f"{g['index']:4d}  {g['name']:24s}  {g['bus']:5d}"
              f"  {g['P_MW']:10.4f}  {g['Pmin']:8.2f}  {g['Pmax']:8.2f}"
              f"  {qg_str}  {g['Pcost']:10.6f}")

    # Load summary
    print_separator("LOAD SUMMARY")
    print(f"{'#':>4s}  {'Name':24s}  {'Bus':>5s}  {'P(MW)':>10s}  {'Q(Mvar)':>10s}")
    print("-" * 60)
    for ld in rd["loads"]:
        print(f"{ld['index']:4d}  {ld['name']:24s}  {ld['bus']:5d}  {ld['P_MW']:10.4f}  {ld['Q_Mvar']:10.4f}")

    # System summary
    s = rd["summary"]
    print_separator("SYSTEM SUMMARY")
    print(f"  Total generation:   P = {s['total_gen_P_MW']:10.4f} MW,  Q = {s['total_gen_Q_Mvar']:10.4f} Mvar")
    print(f"  Total load:         P = {s['total_load_P_MW']:10.4f} MW,  Q = {s['total_load_Q_Mvar']:10.4f} Mvar")
    print(f"  Total losses:       P = {s['total_Ploss_MW']:10.4f} MW,  Q = {s['total_Qloss_Mvar']:10.4f} Mvar")
    print(f"  Total gen cost:       {s['total_cost']:10.6f}")
    mismatch = s["total_gen_P_MW"] - s["total_load_P_MW"] - s["total_Ploss_MW"]
    print(f"  Mismatch (gen-load-loss): P = {mismatch:.4f} MW")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="MCQ Builder → VeraGrid AC OPF (Full Results + JSON Export)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--url",
        default=DEFAULT_PAGE_URL,
        help="GitHub Pages URL of the MCQ Builder app (default: %(default)s)",
    )
    src.add_argument("--json-file", help="Local JSON file with GENERATORS, LOADS, LINES")
    src.add_argument("--json-url", help="URL of a raw JSON model")
    ap.add_argument("--vnom", type=float, default=20.0, help="Nominal line voltage in kV (default 20)")
    ap.add_argument(
        "--results-output", "-o",
        default="opf_results.json",
        help="Path for the structured JSON results file (default: opf_results.json)",
    )
    args = ap.parse_args()

    # ---- Load model ----
    if args.json_file:
        data = load_model_json(args.json_file)
    elif args.json_url:
        data = fetch_model_json_url(args.json_url)
    else:
        data = fetch_model_from_page(args.url)

    grid = build_multicircuit(data, vnom_kv=args.vnom)

    # ---- Run power flow first (for OPF initialization) ----
    pf_opt = vg.PowerFlowOptions(control_q=False)
    pf_drv = vg.PowerFlowDriver(grid=grid, options=pf_opt)
    pf_drv.run()

    # ---- Run AC OPF ----
    opf_opt = vg.OptimalPowerFlowOptions(
        ips_method=vg.SolverType.NR,
        ips_tolerance=1e-8,
        ips_iterations=80,
        acopf_mode=vg.AcOpfMode.ACOPFslacks,
        ips_init_with_pf=True,
        acopf_v0=pf_drv.results.voltage,
        acopf_S0=pf_drv.results.Sbus,
    )
    res = vg.nonlinear_opf(grid=grid, opf_options=opf_opt)

    # ---- Extract structured results ----
    results_dict = extract_results_dict(res, grid)

    # ---- Print to console ----
    print_full_results(results_dict)

    # ---- Export JSON ----
    with open(args.results_output, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n[SAVED] {args.results_output}")
    print(f"        → Feed this into: python opf_mcq_generator_150.py --input {args.results_output}")

    return 0 if results_dict["converged"] else 1


if __name__ == "__main__":
    sys.exit(main())