#!/usr/bin/env python3
"""
opf_mcq_generator_context_free.py
=================================

Deterministic MCQ generator for AC-OPF / power-flow results.
Generates context-free question stems such as:

    "For Line 1-2, what is the active power loss?"
    "For Bus 3, what is the voltage angle?"
    "For Generator 2, what is the incremental cost?"

All formulas, substitutions, and numeric derivations are moved into the
`explanation` field rather than the question stem.

Difficulty tiers:
  - Easy   : direct lookup / counting / identification
  - Medium : one-step derivation / ranking / per-unit / apparent power
  - Hard   : multi-step computation / cross-table inference / statistics /
             reserve / cost share / current magnitude / loss contribution

Input:  opf_results.json
Output: mcq_questions.json

Usage:
  python opf_mcq_generator_context_free.py --input opf_results.json
  python opf_mcq_generator_context_free.py --input opf_results.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Sequence


# ============================================================================
# Data models
# ============================================================================

@dataclass(frozen=True)
class MCQ:
    id: int
    question: str
    options: Dict[str, str]
    correct_answer: str
    correct_value: str
    category: str
    difficulty: str
    explanation: str
    source: str = "deterministic-template"
    template_name: str = ""


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    difficulty: str
    category: str
    generator: Callable[[dict, random.Random, List[int]], List[MCQ]]


# ============================================================================
# Validation
# ============================================================================

def validate_results_schema(results: dict) -> None:
    required = ["Sbase_MVA", "buses", "branches", "generators", "loads", "summary"]
    for k in required:
        if k not in results:
            raise ValueError(f"Missing key: {k}")
    for k in ["buses", "branches", "generators", "loads"]:
        if not isinstance(results[k], list):
            raise ValueError(f"results['{k}'] must be a list")


# ============================================================================
# Utilities
# ============================================================================

LABELS = ["A", "B", "C", "D", "E", "F"]


def r4(x: float) -> float:
    return round(float(x), 4)


def r6(x: float) -> float:
    return round(float(x), 6)


def fmt(x: float, d: int = 4) -> str:
    return f"{float(x):.{d}f}"


def fmu(x: float, unit: str, d: int = 4) -> str:
    return f"{fmt(x, d)} {unit}" if unit else fmt(x, d)


def safe_mean(vals) -> float:
    vals = list(vals)
    return statistics.mean(vals) if vals else 0.0


def safe_stdev(vals) -> float:
    vals = list(vals)
    return statistics.pstdev(vals) if len(vals) >= 2 else 0.0


def nid(counter: List[int]) -> int:
    counter[0] += 1
    return counter[0]


def bname(b: dict) -> str:
    return str(b.get("name", f"Bus {b.get('index', '?')}"))


def gname(g: dict) -> str:
    return str(g.get("name", f"Generator {g.get('index', '?')}"))


def brname(br: dict) -> str:
    return str(br.get("name", f"Line {br.get('index', '?')}"))


def lname(ld: dict) -> str:
    return str(ld.get("name", f"Load {ld.get('index', '?')}"))


def choose_wrong(items: list, correct_name: str, key: str = "name", k: int = 3) -> List[str]:
    out = []
    for item in items:
        v = str(item.get(key, ""))
        if v != str(correct_name) and v not in out:
            out.append(v)
        if len(out) == k:
            break
    return out


def numeric_dist(
    correct: float,
    rng: random.Random,
    n: int = 3,
    spread: float = 0.12,
    floor: float = 1.0,
    d: int = 4,
) -> List[float]:
    out = set()
    mag = max(abs(correct), floor)
    strats = [
        lambda c: c + rng.uniform(-spread, spread) * mag,
        lambda c: c * rng.uniform(0.85, 1.15),
        lambda c: c + rng.choice([-2, -1, 1, 2]) * 0.1 * mag,
        lambda c: -c + rng.uniform(-0.05, 0.05) * mag,
        lambda c: c * rng.choice([0.5, 2.0, 0.9, 1.1]),
    ]
    attempts = 0
    while len(out) < n and attempts < 300:
        attempts += 1
        cand = round(rng.choice(strats)(correct), d)
        if abs(cand - correct) > 10 ** (-d):
            out.add(cand)

    while len(out) < n:
        out.add(round(correct + rng.uniform(0.5, 2.0), d))

    return list(out)[:n]


def make_mcq(
    *,
    qid: int,
    question: str,
    correct: str,
    distractors: Sequence[str],
    category: str,
    difficulty: str,
    explanation: str,
    rng: random.Random,
    template_name: str,
) -> MCQ:
    options = [correct] + list(distractors)
    rng.shuffle(options)
    ci = options.index(correct)
    return MCQ(
        id=qid,
        question=question,
        options={LABELS[i]: options[i] for i in range(len(options))},
        correct_answer=LABELS[ci],
        correct_value=correct,
        category=category,
        difficulty=difficulty,
        explanation=explanation,
        template_name=template_name,
    )


def ask_for_bus(bus: dict, quantity: str) -> str:
    return f"For {bname(bus)}, what is {quantity}?"


def ask_for_gen(gen: dict, quantity: str) -> str:
    return f"For {gname(gen)}, what is {quantity}?"


def ask_for_branch(br: dict, quantity: str) -> str:
    return f"For {brname(br)}, what is {quantity}?"


def ask_for_load(ld: dict, quantity: str) -> str:
    return f"For {lname(ld)}, what is {quantity}?"


# ============================================================================
# EASY TEMPLATES
# ============================================================================

def t_count_elements(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for label, key in [
        ("buses", "buses"),
        ("transmission lines", "branches"),
        ("generators", "generators"),
        ("loads", "loads"),
    ]:
        n = len(R[key])
        qs.append(
            make_mcq(
                qid=nid(C),
                question=f"How many {label} are in the system?",
                correct=str(n),
                distractors=[str(max(n - 1, 0)), str(n + 1), str(n + 2)],
                category="Network Topology",
                difficulty="Easy",
                explanation=f"The system contains {n} {label}.",
                rng=rng,
                template_name="count_elements",
            )
        )
    return qs


def t_extrema_id(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    buses = R["buses"]
    brs = R["branches"]
    gens = R["generators"]

    if len(buses) >= 2:
        b = max(buses, key=lambda x: float(x["Vm"]))
        qs.append(
            make_mcq(
                qid=nid(C),
                question="Which bus has the highest voltage magnitude?",
                correct=bname(b),
                distractors=choose_wrong(buses, bname(b)),
                category="Bus Voltage",
                difficulty="Easy",
                explanation=f"{bname(b)} has the highest voltage magnitude, Vm = {b['Vm']} p.u.",
                rng=rng,
                template_name="extrema_id",
            )
        )

        b = min(buses, key=lambda x: float(x["Vm"]))
        qs.append(
            make_mcq(
                qid=nid(C),
                question="Which bus has the lowest voltage magnitude?",
                correct=bname(b),
                distractors=choose_wrong(buses, bname(b)),
                category="Bus Voltage",
                difficulty="Easy",
                explanation=f"{bname(b)} has the lowest voltage magnitude, Vm = {b['Vm']} p.u.",
                rng=rng,
                template_name="extrema_id",
            )
        )

    if len(brs) >= 2:
        br = max(brs, key=lambda x: float(x["loading_pct"]))
        qs.append(
            make_mcq(
                qid=nid(C),
                question="Which line has the highest loading percentage?",
                correct=brname(br),
                distractors=choose_wrong(brs, brname(br)),
                category="Branch Loading",
                difficulty="Easy",
                explanation=f"{brname(br)} has the highest loading, {br['loading_pct']}%.",
                rng=rng,
                template_name="extrema_id",
            )
        )

    if len(gens) >= 2:
        g = max(gens, key=lambda x: float(x["P_MW"]))
        qs.append(
            make_mcq(
                qid=nid(C),
                question="Which generator has the highest active power dispatch?",
                correct=gname(g),
                distractors=choose_wrong(gens, gname(g)),
                category="Generator Dispatch",
                difficulty="Easy",
                explanation=f"{gname(g)} has the highest dispatch, P = {g['P_MW']} MW.",
                rng=rng,
                template_name="extrema_id",
            )
        )

    return qs


def t_direct_lookup_bus(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for bus in R["buses"]:
        vm = float(bus["Vm"])
        va = float(bus["Va_deg"])

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_bus(bus, "the voltage magnitude"),
                correct=fmu(vm, "p.u."),
                distractors=[fmu(x, "p.u.") for x in numeric_dist(vm, rng, spread=0.03)],
                category="Bus Voltage",
                difficulty="Easy",
                explanation=f"{bname(bus)} has voltage magnitude Vm = {fmu(vm, 'p.u.')}.",
                rng=rng,
                template_name="direct_lookup_bus",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_bus(bus, "the voltage angle"),
                correct=fmu(va, "°"),
                distractors=[fmu(x, "°") for x in numeric_dist(va, rng, spread=0.20)],
                category="Bus Voltage Angle",
                difficulty="Easy",
                explanation=f"{bname(bus)} has voltage angle Va = {fmu(va, '°')}.",
                rng=rng,
                template_name="direct_lookup_bus",
            )
        )

    return qs


def t_direct_lookup_gen(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for gen in R["generators"]:
        p = float(gen["P_MW"])
        q = float(gen.get("Q_Mvar", gen.get("Q", 0.0)))
        cost = float(gen.get("Cost", 0.0))

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the active power dispatch"),
                correct=fmu(p, "MW"),
                distractors=[fmu(x, "MW") for x in numeric_dist(p, rng)],
                category="Generator Dispatch",
                difficulty="Easy",
                explanation=f"{gname(gen)} dispatches active power P = {fmu(p, 'MW')}.",
                rng=rng,
                template_name="direct_lookup_gen",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the linear cost coefficient"),
                correct=fmt(cost, 6),
                distractors=[fmt(x, 6) for x in numeric_dist(cost, rng, d=6, spread=0.15, floor=0.01)],
                category="Economic Dispatch",
                difficulty="Easy",
                explanation=f"{gname(gen)} has linear cost coefficient b = {fmt(cost, 6)}.",
                rng=rng,
                template_name="direct_lookup_gen",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the reactive power dispatch"),
                correct=fmu(q, "Mvar"),
                distractors=[fmu(x, "Mvar") for x in numeric_dist(q, rng)],
                category="Generator Dispatch",
                difficulty="Easy",
                explanation=f"{gname(gen)} dispatches reactive power Q = {fmu(q, 'Mvar')}.",
                rng=rng,
                template_name="direct_lookup_gen",
            )
        )

    return qs


def t_direct_lookup_branch(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for br in R["branches"]:
        pf = float(br["Pf_MW"])
        qf = float(br["Qf_Mvar"])
        ld = float(br["loading_pct"])
        pl = float(br["Ploss_MW"])

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the from-end active power flow"),
                correct=fmu(pf, "MW"),
                distractors=[fmu(x, "MW") for x in numeric_dist(pf, rng)],
                category="Branch Flow",
                difficulty="Easy",
                explanation=f"{brname(br)} has from-end active power flow Pf = {fmu(pf, 'MW')}.",
                rng=rng,
                template_name="direct_lookup_branch",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the from-end reactive power flow"),
                correct=fmu(qf, "Mvar"),
                distractors=[fmu(x, "Mvar") for x in numeric_dist(qf, rng)],
                category="Branch Flow",
                difficulty="Easy",
                explanation=f"{brname(br)} has from-end reactive power flow Qf = {fmu(qf, 'Mvar')}.",
                rng=rng,
                template_name="direct_lookup_branch",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the loading percentage"),
                correct=fmu(ld, "%", 2),
                distractors=[fmu(x, "%", 2) for x in numeric_dist(ld, rng, d=2)],
                category="Branch Loading",
                difficulty="Easy",
                explanation=f"{brname(br)} has loading = {fmu(ld, '%', 2)}.",
                rng=rng,
                template_name="direct_lookup_branch",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the active power loss"),
                correct=fmu(pl, "MW"),
                distractors=[fmu(x, "MW") for x in numeric_dist(pl, rng)],
                category="Branch Losses",
                difficulty="Easy",
                explanation=f"{brname(br)} has active power loss Ploss = {fmu(pl, 'MW')}.",
                rng=rng,
                template_name="direct_lookup_branch",
            )
        )

    return qs


def t_direct_lookup_load(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for ld in R["loads"]:
        p = float(ld["P_MW"])
        q = float(ld["Q_Mvar"])

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_load(ld, "the active power demand"),
                correct=fmu(p, "MW"),
                distractors=[fmu(x, "MW") for x in numeric_dist(p, rng)],
                category="Load Data",
                difficulty="Easy",
                explanation=f"{lname(ld)} has active demand P = {fmu(p, 'MW')}.",
                rng=rng,
                template_name="direct_lookup_load",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_load(ld, "the reactive power demand"),
                correct=fmu(q, "Mvar"),
                distractors=[fmu(x, "Mvar") for x in numeric_dist(q, rng)],
                category="Load Data",
                difficulty="Easy",
                explanation=f"{lname(ld)} has reactive demand Q = {fmu(q, 'Mvar')}.",
                rng=rng,
                template_name="direct_lookup_load",
            )
        )

    return qs


def t_summary_lookup(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    s = R["summary"]
    items = [
        ("What is the total active power generation?", "total_gen_P_MW", "MW", "System Summary"),
        ("What is the total reactive power generation?", "total_gen_Q_Mvar", "Mvar", "System Summary"),
        ("What is the total active power load?", "total_load_P_MW", "MW", "System Summary"),
        ("What is the total reactive power load?", "total_load_Q_Mvar", "Mvar", "System Summary"),
        ("What is the total active power loss?", "total_Ploss_MW", "MW", "System Losses"),
        ("What is the total reactive power loss?", "total_Qloss_Mvar", "Mvar", "System Losses"),
        ("What is the total generation cost?", "total_cost", "", "Generation Cost"),
    ]

    for prompt, key, unit, cat in items:
        v = float(s[key])
        correct = fmu(v, unit) if unit else fmt(v, 6)
        wrong = [fmu(x, unit) if unit else fmt(x, 6) for x in numeric_dist(v, rng, spread=0.12, d=6 if not unit else 4)]
        qs.append(
            make_mcq(
                qid=nid(C),
                question=prompt,
                correct=correct,
                distractors=wrong,
                category=cat,
                difficulty="Easy",
                explanation=f"From the summary, {key} = {correct}.",
                rng=rng,
                template_name="summary_lookup",
            )
        )

    return qs


def t_system_info(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    conv = bool(R.get("converged", True))
    iters = int(R.get("iterations", 0))
    sb = float(R["Sbase_MVA"])

    qs.append(
        make_mcq(
            qid=nid(C),
            question="Did the AC-OPF solver converge?",
            correct="Yes" if conv else "No",
            distractors=["No", "Unknown", "Partially"] if conv else ["Yes", "Unknown", "Partially"],
            category="Solver Info",
            difficulty="Easy",
            explanation=f"The solver convergence flag is {'True' if conv else 'False'}.",
            rng=rng,
            template_name="system_info",
        )
    )

    qs.append(
        make_mcq(
            qid=nid(C),
            question="How many iterations did the solver take?",
            correct=str(iters),
            distractors=[str(max(iters - 3, 0)), str(iters + 5), str(iters + 10)],
            category="Solver Info",
            difficulty="Easy",
            explanation=f"The solver completed in {iters} iterations.",
            rng=rng,
            template_name="system_info",
        )
    )

    qs.append(
        make_mcq(
            qid=nid(C),
            question="What is the system base power?",
            correct=fmu(sb, "MVA"),
            distractors=[fmu(sb / 10, "MVA"), fmu(sb / 2, "MVA"), fmu(sb * 10, "MVA")],
            category="System Parameters",
            difficulty="Easy",
            explanation=f"The base apparent power is Sbase = {fmu(sb, 'MVA')}.",
            rng=rng,
            template_name="system_info",
        )
    )

    return qs


# ============================================================================
# MEDIUM TEMPLATES
# ============================================================================

def t_power_balance(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    s = R["summary"]

    p_load = float(s["total_load_P_MW"])
    p_loss = float(s["total_Ploss_MW"])
    p_gen = r4(p_load + p_loss)

    qs.append(
        make_mcq(
            qid=nid(C),
            question="What is the total active power generation implied by power balance?",
            correct=fmu(p_gen, "MW"),
            distractors=[
                fmu(p_load, "MW"),
                fmu(p_load - p_loss, "MW"),
                fmu(p_load + 2 * p_loss, "MW"),
            ],
            category="Power Balance",
            difficulty="Medium",
            explanation=(
                f"Power balance requires Pgen = Pload + Ploss = "
                f"{fmt(p_load)} + {fmt(p_loss)} = {fmt(p_gen)} MW."
            ),
            rng=rng,
            template_name="power_balance",
        )
    )

    q_load = float(s["total_load_Q_Mvar"])
    q_loss = float(s["total_Qloss_Mvar"])
    q_gen = r4(q_load + q_loss)

    qs.append(
        make_mcq(
            qid=nid(C),
            question="What is the total reactive power generation implied by power balance?",
            correct=fmu(q_gen, "Mvar"),
            distractors=[
                fmu(q_load, "Mvar"),
                fmu(q_load - q_loss, "Mvar"),
                fmu(q_load + 2 * q_loss, "Mvar"),
            ],
            category="Reactive Power Balance",
            difficulty="Medium",
            explanation=(
                f"Reactive power balance requires Qgen = Qload + Qloss = "
                f"{fmt(q_load)} + {fmt(q_loss)} = {fmt(q_gen)} Mvar."
            ),
            rng=rng,
            template_name="power_balance",
        )
    )

    return qs


def t_per_unit_gen(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    sb = float(R["Sbase_MVA"])

    for gen in R["generators"]:
        p = float(gen["P_MW"])
        pu = r6(p / sb)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the per-unit active power output"),
                correct=fmu(pu, "p.u.", 6),
                distractors=[
                    fmu(p, "p.u.", 6),
                    fmu(pu * 10, "p.u.", 6),
                    fmu(pu / 2, "p.u.", 6),
                ],
                category="Per-Unit System",
                difficulty="Medium",
                explanation=(
                    f"Per-unit output = P / Sbase = {fmt(p)} / {fmt(sb)} = "
                    f"{fmt(pu, 6)} p.u."
                ),
                rng=rng,
                template_name="per_unit_gen",
            )
        )

    return qs


def t_per_unit_load(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    sb = float(R["Sbase_MVA"])

    for ld in R["loads"]:
        p = float(ld["P_MW"])
        pu = r6(p / sb)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_load(ld, "the per-unit active power demand"),
                correct=fmu(pu, "p.u.", 6),
                distractors=[
                    fmu(p, "p.u.", 6),
                    fmu(pu * 10, "p.u.", 6),
                    fmu(pu / 2, "p.u.", 6),
                ],
                category="Per-Unit System",
                difficulty="Medium",
                explanation=(
                    f"Per-unit demand = P / Sbase = {fmt(p)} / {fmt(sb)} = "
                    f"{fmt(pu, 6)} p.u."
                ),
                rng=rng,
                template_name="per_unit_load",
            )
        )

    return qs


def t_line_impedance(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for br in R["branches"]:
        rv = float(br["r"])
        xv = float(br["x"])
        z = r6(math.sqrt(rv**2 + xv**2))

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the impedance magnitude"),
                correct=fmu(z, "p.u.", 6),
                distractors=[
                    fmu(rv + xv, "p.u.", 6),
                    fmu(abs(rv - xv), "p.u.", 6),
                    fmu(2 * z, "p.u.", 6),
                ],
                category="Line Parameters",
                difficulty="Medium",
                explanation=(
                    f"|Z| = sqrt(r² + x²) = sqrt({fmt(rv, 6)}² + {fmt(xv, 6)}²) "
                    f"= {fmt(z, 6)} p.u."
                ),
                rng=rng,
                template_name="line_impedance",
            )
        )

    return qs


def t_apparent_power_branch(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for br in R["branches"]:
        pf = float(br["Pf_MW"])
        qf = float(br["Qf_Mvar"])
        s = r4(math.sqrt(pf**2 + qf**2))

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the from-end apparent power"),
                correct=fmu(s, "MVA"),
                distractors=[
                    fmu(abs(pf) + abs(qf), "MVA"),
                    fmu(abs(pf - qf), "MVA"),
                    fmu(1.1 * s, "MVA"),
                ],
                category="Branch Flow",
                difficulty="Medium",
                explanation=(
                    f"|Sf| = sqrt(Pf² + Qf²) = sqrt({fmt(pf)}² + {fmt(qf)}²) "
                    f"= {fmt(s)} MVA."
                ),
                rng=rng,
                template_name="apparent_power_branch",
            )
        )

    return qs


def t_power_factor_branch(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for br in R["branches"]:
        pf = float(br["Pf_MW"])
        qf = float(br["Qf_Mvar"])
        sf = math.sqrt(pf**2 + qf**2)
        if sf < 1e-9:
            continue
        pf_val = r4(abs(pf) / sf)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the from-end power factor"),
                correct=fmt(pf_val),
                distractors=[fmt(x) for x in numeric_dist(pf_val, rng, spread=0.12, floor=0.1)],
                category="Power Factor",
                difficulty="Medium",
                explanation=(
                    f"Power factor = |P| / |S| = {fmt(abs(pf))} / {fmt(sf)} = "
                    f"{fmt(pf_val)}."
                ),
                rng=rng,
                template_name="power_factor_branch",
            )
        )

    return qs


def t_gen_utilization(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for gen in R["generators"]:
        p = float(gen["P_MW"])
        pmax = float(gen["Pmax"])
        if pmax <= 0:
            continue
        util = r4(100.0 * p / pmax)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the utilization percentage"),
                correct=fmu(util, "%"),
                distractors=[fmu(x, "%") for x in numeric_dist(util, rng, spread=0.15)],
                category="Generator Utilization",
                difficulty="Medium",
                explanation=(
                    f"Utilization = (P / Pmax) × 100 = ({fmt(p)} / {fmt(pmax)}) × 100 "
                    f"= {fmt(util)}%."
                ),
                rng=rng,
                template_name="gen_utilization",
            )
        )

    return qs


def t_gen_reserve(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for gen in R["generators"]:
        p = float(gen["P_MW"])
        pmax = float(gen["Pmax"])
        reserve = r4(pmax - p)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the available reserve"),
                correct=fmu(reserve, "MW"),
                distractors=[
                    fmu(p, "MW"),
                    fmu(pmax, "MW"),
                    fmu(reserve / 2, "MW"),
                ],
                category="Generator Reserve",
                difficulty="Medium",
                explanation=(
                    f"Reserve = Pmax - P = {fmt(pmax)} - {fmt(p)} = {fmt(reserve)} MW."
                ),
                rng=rng,
                template_name="gen_reserve",
            )
        )

    return qs


def t_rx_ratio(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for br in R["branches"]:
        rv = float(br["r"])
        xv = float(br["x"])
        if abs(xv) < 1e-12:
            continue
        rx = r4(rv / xv)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the R/X ratio"),
                correct=fmt(rx),
                distractors=[fmt(x) for x in numeric_dist(rx, rng, spread=0.18, floor=0.01)],
                category="Line Parameters",
                difficulty="Medium",
                explanation=(
                    f"R/X = r / x = {fmt(rv, 6)} / {fmt(xv, 6)} = {fmt(rx)}."
                ),
                rng=rng,
                template_name="rx_ratio",
            )
        )

    return qs


def t_loss_ratio(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    s = R["summary"]
    pgen = float(s["total_gen_P_MW"])
    ploss = float(s["total_Ploss_MW"])
    if pgen <= 0:
        return qs

    loss_ratio = r4(100.0 * ploss / pgen)

    qs.append(
        make_mcq(
            qid=nid(C),
            question="What percentage of total generation is lost in the system?",
            correct=fmu(loss_ratio, "%"),
            distractors=[fmu(x, "%") for x in numeric_dist(loss_ratio, rng, spread=0.15)],
            category="System Losses",
            difficulty="Medium",
            explanation=(
                f"Loss ratio = (Ploss / Pgen) × 100 = ({fmt(ploss)} / {fmt(pgen)}) × 100 "
                f"= {fmt(loss_ratio)}%."
            ),
            rng=rng,
            template_name="loss_ratio",
        )
    )

    return qs


# ============================================================================
# HARD TEMPLATES
# ============================================================================

def t_cross_table_loss(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for br in R["branches"]:
        pf = float(br["Pf_MW"])
        pt = float(br["Pt_MW"])
        qf = float(br["Qf_Mvar"])
        qt = float(br["Qt_Mvar"])

        pl = r4(pf + pt)
        ql = r4(qf + qt)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the active power loss"),
                correct=fmu(pl, "MW"),
                distractors=[
                    fmu(pf - pt, "MW"),
                    fmu(abs(pf) + abs(pt), "MW"),
                    fmu(-pl, "MW"),
                ],
                category="Branch Losses",
                difficulty="Hard",
                explanation=(
                    f"Active loss is computed from both ends: Ploss = Pf + Pt = "
                    f"{fmt(pf)} + {fmt(pt)} = {fmt(pl)} MW."
                ),
                rng=rng,
                template_name="cross_table_loss",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the reactive power loss"),
                correct=fmu(ql, "Mvar"),
                distractors=[
                    fmu(qf - qt, "Mvar"),
                    fmu(abs(qf) + abs(qt), "Mvar"),
                    fmu(-ql, "Mvar"),
                ],
                category="Branch Losses",
                difficulty="Hard",
                explanation=(
                    f"Reactive loss is computed from both ends: Qloss = Qf + Qt = "
                    f"{fmt(qf)} + {fmt(qt)} = {fmt(ql)} Mvar."
                ),
                rng=rng,
                template_name="cross_table_loss",
            )
        )

    return qs


def t_margin_to_limit(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    brs = sorted(R["branches"], key=lambda x: float(x["loading_pct"]), reverse=True)
    if not brs:
        return qs

    critical = brs[0]
    loading = float(critical["loading_pct"])
    margin = r4(100.0 - loading)

    qs.append(
        make_mcq(
            qid=nid(C),
            question="Which line is closest to its thermal limit?",
            correct=brname(critical),
            distractors=[brname(b) for b in brs[1:4]] if len(brs) >= 4 else [brname(b) for b in brs[1:]],
            category="Constraint Analysis",
            difficulty="Hard",
            explanation=(
                f"The line closest to its limit is the one with the highest loading. "
                f"{brname(critical)} has loading = {fmt(loading)}%."
            ),
            rng=rng,
            template_name="margin_to_limit",
        )
    )

    qs.append(
        make_mcq(
            qid=nid(C),
            question=ask_for_branch(critical, "the remaining thermal margin"),
            correct=fmu(margin, "%"),
            distractors=[
                fmu(loading, "%"),
                fmu(margin / 2, "%"),
                fmu(margin * 2, "%"),
            ],
            category="Constraint Analysis",
            difficulty="Hard",
            explanation=(
                f"Remaining thermal margin = 100% - loading = 100 - {fmt(loading)} = "
                f"{fmt(margin)}%."
            ),
            rng=rng,
            template_name="margin_to_limit",
        )
    )

    return qs


def t_economic_reasoning(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    gens = R["generators"]
    if len(gens) < 2:
        return qs

    cheapest = min(gens, key=lambda g: float(g["Cost"]))
    expensive = max(gens, key=lambda g: float(g["Cost"]))

    wrong1 = [gname(g) for g in gens if gname(g) != gname(cheapest)][:3]
    wrong2 = [gname(g) for g in gens if gname(g) != gname(expensive)][:3]

    qs.append(
        make_mcq(
            qid=nid(C),
            question="If total load increases and no limits bind, which generator should increase output first?",
            correct=gname(cheapest),
            distractors=wrong1,
            category="Economic Dispatch",
            difficulty="Hard",
            explanation=(
                f"Under economic dispatch, the cheapest available generator is dispatched first. "
                f"{gname(cheapest)} has the lowest linear cost coefficient, {fmt(float(cheapest['Cost']), 6)}."
            ),
            rng=rng,
            template_name="economic_reasoning",
        )
    )

    qs.append(
        make_mcq(
            qid=nid(C),
            question="If total load decreases, which generator should reduce output first?",
            correct=gname(expensive),
            distractors=wrong2,
            category="Economic Dispatch",
            difficulty="Hard",
            explanation=(
                f"Under economic dispatch, the most expensive generator is backed down first. "
                f"{gname(expensive)} has the highest linear cost coefficient, {fmt(float(expensive['Cost']), 6)}."
            ),
            rng=rng,
            template_name="economic_reasoning",
        )
    )

    return qs


def t_system_statistics(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    buses = R["buses"]
    brs = R["branches"]

    if len(buses) >= 2:
        vms = [float(b["Vm"]) for b in buses]
        avg_vm = r4(safe_mean(vms))
        std_vm = r6(safe_stdev(vms))
        spread_vm = r6(max(vms) - min(vms))

        qs.append(
            make_mcq(
                qid=nid(C),
                question="What is the average bus voltage magnitude?",
                correct=fmu(avg_vm, "p.u."),
                distractors=[
                    fmu(min(vms), "p.u."),
                    fmu(max(vms), "p.u."),
                    fmu(avg_vm + 0.05, "p.u."),
                ],
                category="System Statistics",
                difficulty="Hard",
                explanation=f"Average Vm is the arithmetic mean of all bus magnitudes, equal to {fmu(avg_vm, 'p.u.')}.",
                rng=rng,
                template_name="system_statistics",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question="What is the standard deviation of bus voltage magnitudes?",
                correct=fmu(std_vm, "p.u.", 6),
                distractors=[fmu(x, "p.u.", 6) for x in numeric_dist(std_vm, rng, d=6, spread=0.25, floor=0.001)],
                category="System Statistics",
                difficulty="Hard",
                explanation=f"The population standard deviation of bus voltage magnitudes is {fmu(std_vm, 'p.u.', 6)}.",
                rng=rng,
                template_name="system_statistics",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question="What is the voltage spread across all buses?",
                correct=fmu(spread_vm, "p.u.", 6),
                distractors=[fmu(x, "p.u.", 6) for x in numeric_dist(spread_vm, rng, d=6, spread=0.25, floor=0.001)],
                category="System Statistics",
                difficulty="Hard",
                explanation=(
                    f"Voltage spread = max(Vm) - min(Vm) = "
                    f"{fmt(max(vms), 6)} - {fmt(min(vms), 6)} = {fmt(spread_vm, 6)} p.u."
                ),
                rng=rng,
                template_name="system_statistics",
            )
        )

    if len(brs) >= 2:
        losses = [float(br["Ploss_MW"]) for br in brs]
        loadings = [float(br["loading_pct"]) for br in brs]
        avg_loss = r4(safe_mean(losses))
        avg_loading = r4(safe_mean(loadings))

        qs.append(
            make_mcq(
                qid=nid(C),
                question="What is the average active power loss per line?",
                correct=fmu(avg_loss, "MW"),
                distractors=[
                    fmu(sum(losses), "MW"),
                    fmu(max(losses), "MW"),
                    fmu(avg_loss * 2, "MW"),
                ],
                category="System Statistics",
                difficulty="Hard",
                explanation=f"The average active power loss per line is {fmu(avg_loss, 'MW')}.",
                rng=rng,
                template_name="system_statistics",
            )
        )

        qs.append(
            make_mcq(
                qid=nid(C),
                question="What is the average loading percentage across all lines?",
                correct=fmu(avg_loading, "%"),
                distractors=[fmu(x, "%") for x in numeric_dist(avg_loading, rng, spread=0.15)],
                category="System Statistics",
                difficulty="Hard",
                explanation=f"The average line loading is {fmu(avg_loading, '%')}.",
                rng=rng,
                template_name="system_statistics",
            )
        )

    return qs


def t_apparent_power_loss(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for br in R["branches"]:
        pl = float(br["Ploss_MW"])
        ql = float(br["Qloss_Mvar"])
        sl = r4(math.sqrt(pl**2 + ql**2))

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the apparent power loss"),
                correct=fmu(sl, "MVA"),
                distractors=[
                    fmu(abs(pl) + abs(ql), "MVA"),
                    fmu(abs(pl - ql), "MVA"),
                    fmu(1.5 * sl, "MVA"),
                ],
                category="Branch Losses",
                difficulty="Hard",
                explanation=(
                    f"|Sloss| = sqrt(Ploss² + Qloss²) = sqrt({fmt(pl)}² + {fmt(ql)}²) "
                    f"= {fmt(sl)} MVA."
                ),
                rng=rng,
                template_name="apparent_power_loss",
            )
        )

    return qs


def t_total_reserve(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    gens = R["generators"]
    if not gens:
        return qs

    total_reserve = r4(sum(float(g["Pmax"]) - float(g["P_MW"]) for g in gens))
    total_pmax = r4(sum(float(g["Pmax"]) for g in gens))

    qs.append(
        make_mcq(
            qid=nid(C),
            question="What is the total spinning reserve in the system?",
            correct=fmu(total_reserve, "MW"),
            distractors=[
                fmu(total_pmax, "MW"),
                fmu(total_reserve / 2, "MW"),
                fmu(total_reserve * 1.5, "MW"),
            ],
            category="System Reserve",
            difficulty="Hard",
            explanation=(
                f"Total reserve = Σ(Pmax - P) over all generators = {fmt(total_reserve)} MW."
            ),
            rng=rng,
            template_name="total_reserve",
        )
    )

    if total_pmax > 0:
        margin = r4(100.0 * total_reserve / total_pmax)
        qs.append(
            make_mcq(
                qid=nid(C),
                question="What is the reserve margin as a percentage of total generator capacity?",
                correct=fmu(margin, "%"),
                distractors=[fmu(x, "%") for x in numeric_dist(margin, rng, spread=0.15)],
                category="System Reserve",
                difficulty="Hard",
                explanation=(
                    f"Reserve margin = (total reserve / total Pmax) × 100 = "
                    f"({fmt(total_reserve)} / {fmt(total_pmax)}) × 100 = {fmt(margin)}%."
                ),
                rng=rng,
                template_name="total_reserve",
            )
        )

    return qs


def t_angle_difference(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    buses_by_idx = {int(b["index"]): b for b in R["buses"]}

    for br in R["branches"]:
        fi = int(br["from_bus"])
        ti = int(br["to_bus"])
        if fi not in buses_by_idx or ti not in buses_by_idx:
            continue

        va_f = float(buses_by_idx[fi]["Va_deg"])
        va_t = float(buses_by_idx[ti]["Va_deg"])
        delta = r4(va_f - va_t)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the bus angle difference"),
                correct=fmu(delta, "°"),
                distractors=[
                    fmu(va_t - va_f, "°"),
                    fmu(abs(delta) * 2, "°"),
                    fmu(delta / 2, "°"),
                ],
                category="Angle Analysis",
                difficulty="Hard",
                explanation=(
                    f"Angle difference is δ = Va_from - Va_to = "
                    f"{fmt(va_f)} - {fmt(va_t)} = {fmt(delta)}°."
                ),
                rng=rng,
                template_name="angle_difference",
            )
        )

    return qs


def t_branch_current_magnitude(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    sb = float(R["Sbase_MVA"])
    buses_by_idx = {int(b["index"]): b for b in R["buses"]}

    for br in R["branches"]:
        fi = int(br["from_bus"])
        if fi not in buses_by_idx:
            continue

        pf_pu = float(br["Pf_MW"]) / sb
        qf_pu = float(br["Qf_Mvar"]) / sb
        sf_pu = math.sqrt(pf_pu**2 + qf_pu**2)

        vm_f = float(buses_by_idx[fi]["Vm"])
        if vm_f < 1e-9:
            continue

        i_mag = r6(sf_pu / vm_f)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the current magnitude"),
                correct=fmu(i_mag, "p.u.", 6),
                distractors=[
                    fmu(sf_pu * vm_f, "p.u.", 6),
                    fmu(sf_pu, "p.u.", 6),
                    fmu(2 * i_mag, "p.u.", 6),
                ],
                category="Branch Current",
                difficulty="Hard",
                explanation=(
                    f"First convert branch power to per-unit. "
                    f"Pf_pu = {fmt(float(br['Pf_MW']), 6)} / {fmt(sb, 6)} = {fmt(pf_pu, 6)} p.u., "
                    f"Qf_pu = {fmt(float(br['Qf_Mvar']), 6)} / {fmt(sb, 6)} = {fmt(qf_pu, 6)} p.u. "
                    f"Then |Sf| = sqrt(Pf_pu² + Qf_pu²) = {fmt(sf_pu, 6)} p.u. "
                    f"Finally, |If| = |Sf| / Vm_from = {fmt(sf_pu, 6)} / {fmt(vm_f, 6)} = {fmt(i_mag, 6)} p.u."
                ),
                rng=rng,
                template_name="branch_current_magnitude",
            )
        )

    return qs


def t_loss_fraction_per_branch(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    total_ploss = sum(float(br["Ploss_MW"]) for br in R["branches"])
    if total_ploss <= 0:
        return qs

    for br in R["branches"]:
        pl = float(br["Ploss_MW"])
        frac = r4(100.0 * pl / total_ploss)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_branch(br, "the percentage contribution to total active power loss"),
                correct=fmu(frac, "%"),
                distractors=[fmu(x, "%") for x in numeric_dist(frac, rng, spread=0.18)],
                category="Loss Distribution",
                difficulty="Hard",
                explanation=(
                    f"Loss share = (branch Ploss / total Ploss) × 100 = "
                    f"({fmt(pl)} / {fmt(total_ploss)}) × 100 = {fmt(frac)}%."
                ),
                rng=rng,
                template_name="loss_fraction_per_branch",
            )
        )

    return qs


def t_gen_cost_per_mw(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for gen in R["generators"]:
        p = float(gen["P_MW"])
        pc = float(gen["Pcost"])
        if p <= 1e-9:
            continue

        cpmw = r6(pc / p)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the average cost per MW"),
                correct=fmt(cpmw, 6),
                distractors=[fmt(x, 6) for x in numeric_dist(cpmw, rng, d=6, spread=0.18, floor=0.001)],
                category="Economic Analysis",
                difficulty="Hard",
                explanation=(
                    f"Average cost per MW = Pcost / P = {fmt(pc, 6)} / {fmt(p)} = {fmt(cpmw, 6)}."
                ),
                rng=rng,
                template_name="gen_cost_per_mw",
            )
        )

    return qs


def t_cost_share_per_gen(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    total_cost = sum(float(g["Pcost"]) for g in R["generators"])
    if total_cost <= 0:
        return qs

    for gen in R["generators"]:
        pc = float(gen["Pcost"])
        share = r4(100.0 * pc / total_cost)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the percentage contribution to total generation cost"),
                correct=fmu(share, "%"),
                distractors=[fmu(x, "%") for x in numeric_dist(share, rng, spread=0.15)],
                category="Cost Distribution",
                difficulty="Hard",
                explanation=(
                    f"Cost share = (generator cost / total cost) × 100 = "
                    f"({fmt(pc, 6)} / {fmt(total_cost, 6)}) × 100 = {fmt(share)}%."
                ),
                rng=rng,
                template_name="cost_share_per_gen",
            )
        )

    return qs


def t_voltage_deviation(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for bus in R["buses"]:
        vm = float(bus["Vm"])
        dev = r6(abs(vm - 1.0))

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_bus(bus, "the voltage deviation from 1.0 p.u."),
                correct=fmu(dev, "p.u.", 6),
                distractors=[fmu(x, "p.u.", 6) for x in numeric_dist(dev, rng, d=6, spread=0.25, floor=0.001)],
                category="Voltage Quality",
                difficulty="Hard",
                explanation=(
                    f"Voltage deviation = |Vm - 1.0| = |{fmt(vm, 6)} - 1.0| = {fmt(dev, 6)} p.u."
                ),
                rng=rng,
                template_name="voltage_deviation",
            )
        )

    return qs


def t_total_apparent_loss(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    s = R["summary"]
    pl = float(s["total_Ploss_MW"])
    ql = float(s["total_Qloss_Mvar"])
    sl = r4(math.sqrt(pl**2 + ql**2))

    qs.append(
        make_mcq(
            qid=nid(C),
            question="What is the total apparent power loss in the system?",
            correct=fmu(sl, "MVA"),
            distractors=[
                fmu(pl + ql, "MVA"),
                fmu(abs(pl - ql), "MVA"),
                fmu(1.5 * sl, "MVA"),
            ],
            category="System Losses",
            difficulty="Hard",
            explanation=(
                f"Total apparent loss = sqrt(Ploss² + Qloss²) = sqrt({fmt(pl)}² + {fmt(ql)}²) "
                f"= {fmt(sl)} MVA."
            ),
            rng=rng,
            template_name="total_apparent_loss",
        )
    )

    return qs


def t_gen_dispatch_share(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    total_pg = sum(float(g["P_MW"]) for g in R["generators"])
    if total_pg <= 0:
        return qs

    for gen in R["generators"]:
        p = float(gen["P_MW"])
        share = r4(100.0 * p / total_pg)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the percentage contribution to total active generation"),
                correct=fmu(share, "%"),
                distractors=[fmu(x, "%") for x in numeric_dist(share, rng, spread=0.15)],
                category="Generation Distribution",
                difficulty="Hard",
                explanation=(
                    f"Generation share = (generator P / total generation) × 100 = "
                    f"({fmt(p)} / {fmt(total_pg)}) × 100 = {fmt(share)}%."
                ),
                rng=rng,
                template_name="gen_dispatch_share",
            )
        )

    return qs


def t_incremental_cost(R: dict, rng: random.Random, C: List[int]) -> List[MCQ]:
    qs = []
    for gen in R["generators"]:
        a = float(gen.get("Cost2", 0.0))
        b = float(gen.get("Cost", 0.0))
        p = float(gen["P_MW"])
        ic = r6(2 * a * p + b)

        qs.append(
            make_mcq(
                qid=nid(C),
                question=ask_for_gen(gen, "the incremental cost"),
                correct=fmt(ic, 6),
                distractors=[fmt(x, 6) for x in numeric_dist(ic, rng, d=6, spread=0.15, floor=0.01)],
                category="Economic Analysis",
                difficulty="Hard",
                explanation=(
                    f"Incremental cost is dC/dP = 2aP + b = "
                    f"2 × {fmt(a, 6)} × {fmt(p)} + {fmt(b, 6)} = {fmt(ic, 6)}."
                ),
                rng=rng,
                template_name="incremental_cost",
            )
        )

    return qs


# ============================================================================
# Template registry
# ============================================================================

EASY_TEMPLATES: List[TemplateSpec] = [
    TemplateSpec("count_elements", "Easy", "Network Topology", t_count_elements),
    TemplateSpec("extrema_id", "Easy", "System Identification", t_extrema_id),
    TemplateSpec("direct_lookup_bus", "Easy", "Bus Voltage", t_direct_lookup_bus),
    TemplateSpec("direct_lookup_gen", "Easy", "Generator Dispatch", t_direct_lookup_gen),
    TemplateSpec("direct_lookup_branch", "Easy", "Branch Flow", t_direct_lookup_branch),
    TemplateSpec("direct_lookup_load", "Easy", "Load Data", t_direct_lookup_load),
    TemplateSpec("summary_lookup", "Easy", "System Summary", t_summary_lookup),
    TemplateSpec("system_info", "Easy", "Solver Info", t_system_info),
]

MEDIUM_TEMPLATES: List[TemplateSpec] = [
    TemplateSpec("power_balance", "Medium", "Power Balance", t_power_balance),
    TemplateSpec("per_unit_gen", "Medium", "Per-Unit System", t_per_unit_gen),
    TemplateSpec("per_unit_load", "Medium", "Per-Unit System", t_per_unit_load),
    TemplateSpec("line_impedance", "Medium", "Line Parameters", t_line_impedance),
    TemplateSpec("apparent_power_branch", "Medium", "Branch Flow", t_apparent_power_branch),
    TemplateSpec("power_factor_branch", "Medium", "Power Factor", t_power_factor_branch),
    TemplateSpec("gen_utilization", "Medium", "Generator Utilization", t_gen_utilization),
    TemplateSpec("gen_reserve", "Medium", "Generator Reserve", t_gen_reserve),
    TemplateSpec("rx_ratio", "Medium", "Line Parameters", t_rx_ratio),
    TemplateSpec("loss_ratio", "Medium", "System Losses", t_loss_ratio),
]

HARD_TEMPLATES: List[TemplateSpec] = [
    TemplateSpec("cross_table_loss", "Hard", "Branch Losses", t_cross_table_loss),
    TemplateSpec("margin_to_limit", "Hard", "Constraint Analysis", t_margin_to_limit),
    TemplateSpec("economic_reasoning", "Hard", "Economic Dispatch", t_economic_reasoning),
    TemplateSpec("system_statistics", "Hard", "System Statistics", t_system_statistics),
    TemplateSpec("apparent_power_loss", "Hard", "Branch Losses", t_apparent_power_loss),
    TemplateSpec("total_reserve", "Hard", "System Reserve", t_total_reserve),
    TemplateSpec("angle_difference", "Hard", "Angle Analysis", t_angle_difference),
    TemplateSpec("branch_current_magnitude", "Hard", "Branch Current", t_branch_current_magnitude),
    TemplateSpec("loss_fraction_per_branch", "Hard", "Loss Distribution", t_loss_fraction_per_branch),
    TemplateSpec("gen_cost_per_mw", "Hard", "Economic Analysis", t_gen_cost_per_mw),
    TemplateSpec("cost_share_per_gen", "Hard", "Cost Distribution", t_cost_share_per_gen),
    TemplateSpec("voltage_deviation", "Hard", "Voltage Quality", t_voltage_deviation),
    TemplateSpec("total_apparent_loss", "Hard", "System Losses", t_total_apparent_loss),
    TemplateSpec("gen_dispatch_share", "Hard", "Generation Distribution", t_gen_dispatch_share),
    TemplateSpec("incremental_cost", "Hard", "Economic Analysis", t_incremental_cost),
]

ALL_TEMPLATES = EASY_TEMPLATES + MEDIUM_TEMPLATES + HARD_TEMPLATES


# ============================================================================
# Assembly / dedupe / cap
# ============================================================================

def dedupe(questions: List[MCQ]) -> List[MCQ]:
    seen = set()
    out = []
    for q in questions:
        key = (q.question.strip(), q.correct_value.strip())
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def cap_questions(questions: List[MCQ], target: int, difficulty: str, rng: random.Random) -> List[MCQ]:
    pool = [q for q in questions if q.difficulty == difficulty]
    if len(pool) <= target:
        return pool
    return rng.sample(pool, target)


def renumber(questions: List[MCQ]) -> List[MCQ]:
    out = []
    for i, q in enumerate(questions, start=1):
        out.append(
            MCQ(
                id=i,
                question=q.question,
                options=q.options,
                correct_answer=q.correct_answer,
                correct_value=q.correct_value,
                category=q.category,
                difficulty=q.difficulty,
                explanation=q.explanation,
                source=q.source,
                template_name=q.template_name,
            )
        )
    return out


def generate_all(
    results: dict,
    seed: int = 0,
    target_easy: int = 50,
    target_medium: int = 50,
    target_hard: int = 50,
) -> List[MCQ]:
    validate_results_schema(results)
    rng = random.Random(seed)
    counter = [0]

    all_qs: List[MCQ] = []
    for spec in ALL_TEMPLATES:
        all_qs.extend(spec.generator(results, rng, counter))

    all_qs = dedupe(all_qs)

    easy = cap_questions(all_qs, target_easy, "Easy", rng)
    medium = cap_questions(all_qs, target_medium, "Medium", rng)
    hard = cap_questions(all_qs, target_hard, "Hard", rng)

    rng.shuffle(easy)
    rng.shuffle(medium)
    rng.shuffle(hard)

    combined = renumber(easy + medium + hard)

    print(
        f"[INFO] Generated {len(easy)} Easy, {len(medium)} Medium, "
        f"{len(hard)} Hard = {len(combined)} total questions."
    )
    return combined


def summarize(questions: List[MCQ]) -> dict:
    diff: Dict[str, int] = {}
    cat: Dict[str, int] = {}
    tmpl: Dict[str, int] = {}

    for q in questions:
        diff[q.difficulty] = diff.get(q.difficulty, 0) + 1
        cat[q.category] = cat.get(q.category, 0) + 1
        tmpl[q.template_name] = tmpl.get(q.template_name, 0) + 1

    return {
        "total_questions": len(questions),
        "difficulty_counts": diff,
        "category_counts": cat,
        "template_counts": tmpl,
    }


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deterministic AC-OPF MCQ generator with context-free stems"
    )
    ap.add_argument("--input", required=True, help="Path to opf_results.json")
    ap.add_argument("--output", "-o", default="mcq_questions.json", help="Output JSON path")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for deterministic option ordering/sampling")
    ap.add_argument("--easy", type=int, default=50, help="Target number of Easy questions")
    ap.add_argument("--medium", type=int, default=50, help="Target number of Medium questions")
    ap.add_argument("--hard", type=int, default=50, help="Target number of Hard questions")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    questions = generate_all(
        results,
        seed=args.seed,
        target_easy=args.easy,
        target_medium=args.medium,
        target_hard=args.hard,
    )

    output = {
        "metadata": {
            "seed": args.seed,
            "requested_easy": args.easy,
            "requested_medium": args.medium,
            "requested_hard": args.hard,
            "generated_easy": sum(1 for q in questions if q.difficulty == "Easy"),
            "generated_medium": sum(1 for q in questions if q.difficulty == "Medium"),
            "generated_hard": sum(1 for q in questions if q.difficulty == "Hard"),
            "total_questions": len(questions),
        },
        "opf_summary": results.get("summary"),
        "summary": summarize(questions),
        "questions": [asdict(q) for q in questions],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[DONE] Wrote {len(questions)} questions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())