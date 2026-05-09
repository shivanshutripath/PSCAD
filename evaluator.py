"""
VeraGrid Power-Flow MCQ Evaluator (v2)
=======================================
Evaluates LLM accuracy on power-flow MCQs in two genuinely different modes:

  no_tool_access:  Pure knowledge — LLM answers from training data only.
                   No tools, no simulation, just pick A/B/C/D.

  agent:           Tool-augmented — LLM has access to VeraGrid tools.
                   It can write inputs, run simulations, read results,
                   and compute the answer before picking A/B/C/D.

Usage:
  # No-tool baseline
  python evaluator.py --model gpt-4o --mode no_tool_access --questions-file mcqs.json

  # Agent with VeraGrid tools
  python evaluator.py --model gpt-4o --mode agent --questions-file mcqs.json

  # Compare both
  python evaluator.py --model gpt-4o --mode agent --questions-file mcqs.json --compare
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalQuestion:
    question_id: int
    question: str
    options: list[tuple[str, str]]
    expected: str
    requires_simulation: bool = False  # tag for analysis
    difficulty: str = "Unknown"
    category: str = ""
    source: str = ""


@dataclass
class EvalResult:
    question_id: int
    question: str
    model_raw: str
    model_answer: str
    expected: str
    correct: bool
    mode: str
    tool_calls: int = 0
    requires_simulation: bool = False
    difficulty: str = "Unknown"
    category: str = ""
    source: str = ""
    evaluation_trace: dict[str, Any] = field(default_factory=dict)


DEFAULT_NO_TOOL_PROMPT = """You are a power systems engineer answering multiple-choice questions.
Answer each question with ONLY the option letter (A, B, C, or D).
Do not include any explanation. Just the letter."""


def canonical_mode(mode: str) -> str:
    """Map mode aliases used by frontend to evaluator canonical modes."""
    m = str(mode or "").strip().lower()
    aliases = {
        "no_tool_access": "no_tool_access",
        "no_tool_use": "no_tool_access",
        "no-tool-use": "no_tool_access",
        "no_tool": "no_tool_access",
        "agent": "agent",
    }
    if m not in aliases:
        raise ValueError(f"Unsupported mode: {mode}")
    return aliases[m]


def extract_letter(text: str) -> str:
    text = (text or "").strip().upper()
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    for ch in text:
        if ch in "ABCD":
            return ch
    return "?"


def normalize_options(options: Any) -> list[tuple[str, str]]:
    if isinstance(options, dict):
        result = []
        for key in sorted(options.keys()):
            letter = str(key).strip().upper()
            if letter in {"A", "B", "C", "D"}:
                result.append((letter, str(options[key])))
        return result
    if isinstance(options, list):
        result = []
        for item in options:
            txt = str(item)
            letter = extract_letter(txt)
            if letter in {"A", "B", "C", "D"}:
                cleaned = re.sub(r"^\s*[A-D][\.\):\-]?\s*", "", txt).strip()
                result.append((letter, cleaned))
        return result
    return []


def load_questions(path: str) -> list[EvalQuestion]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw.get("questions", [])
    if not entries:
        raise ValueError("No questions found.")

    questions = []
    for i, q in enumerate(entries, start=1):
        opts = normalize_options(q.get("options"))
        if len(opts) < 2:
            continue
        qid = int(q.get("id", i))
        qtext = str(q.get("question", "")).strip()
        expected = str(q.get("correct_answer", "")).strip().upper()
        if expected not in {"A", "B", "C", "D"} or not qtext:
            continue
        requires_sim = q.get("requires_simulation", False)
        difficulty = str(q.get("difficulty", "Unknown")).strip() or "Unknown"
        category = str(q.get("category", "")).strip()
        source = str(q.get("source", "")).strip()
        questions.append(EvalQuestion(qid, qtext, opts, expected, requires_sim, difficulty, category, source))

    if not questions:
        raise ValueError("No valid questions parsed.")
    return questions


# ---------------------------------------------------------------------------
# Mode: no_tool_access — pure LLM knowledge
# ---------------------------------------------------------------------------

def _is_openai_reasoning_model(model: str) -> bool:
    """Reasoning families (o-series, gpt-5*) reject `temperature` on the
    Responses API and need extra output budget to fit reasoning tokens.

    Delegates to the shared helper in `agent` so the agent loop and the
    no-tool path stay in sync about which models are reasoning models.
    """
    from agent import is_openai_reasoning_model

    return is_openai_reasoning_model(model)


def _is_cursor_model(model: str) -> bool:
    m = (model or "").lower().strip()
    return m.startswith("cursor:") or m.startswith("cursor-")


def answer_no_tool(
    model: str,
    question: str,
    system_prompt: str | None = None,
    api_key_openai: str | None = None,
    api_key_claude: str | None = None,
    api_key_cursor: str | None = None,
    workspace_base: str | None = None,
    circuit_data: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call the LLM with NO tools. Pure knowledge retrieval.

    OpenAI models go through `client.responses.create` (Responses API) without
    any tools attached. Anthropic models still use `messages.create`. Cursor
    models shell out to the `cursor-agent` CLI in `--mode ask` (read-only) with
    a strong no-tool directive prepended to the prompt.

    Returns ``(raw_text, trace_dict)``. The trace is non-empty only for the
    cursor path (which actually does have tool surfaces available).
    """
    effective_prompt = (system_prompt or "").strip() or DEFAULT_NO_TOOL_PROMPT

    if _is_cursor_model(model):
        from agent import run_cursor_agent

        workspace = tempfile.mkdtemp(prefix="vg_cursor_notool_", dir=workspace_base)
        try:
            response = run_cursor_agent(
                model=model,
                user_prompt=f"{effective_prompt}\n\n{question}",
                workspace=workspace,
                api_key=api_key_cursor,
                circuit_data=circuit_data,
                return_trace=True,
                no_tools=True,
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        if isinstance(response, dict):
            return str(response.get("answer", "")), dict(response.get("trace") or {})
        return str(response), {}

    if model.lower().startswith("claude"):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")
        key = (api_key_claude or "").strip() or os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            system=effective_prompt,
            max_tokens=16,
            messages=[{"role": "user", "content": question}],
        )
        parts = getattr(resp, "content", []) or []
        return "\n".join(getattr(p, "text", "") for p in parts).strip(), {}

    import openai
    key = (api_key_openai or "").strip() or os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=key)

    is_reasoning = _is_openai_reasoning_model(model)
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": effective_prompt,
        "input": question,
        "store": False,
        # Reasoning models also spend output budget on hidden reasoning
        # tokens, so leave plenty of room. The visible answer is still just
        # a single letter and `extract_letter` will pick it out. We don't
        # set `temperature` and let the model use its default.
        "max_output_tokens": 4096 if is_reasoning else 64,
    }

    resp = client.responses.create(**kwargs)

    text = getattr(resp, "output_text", "") or ""
    if not text:
        # Fall back to walking the structured output blocks if the
        # convenience accessor is empty (older SDKs / refusal blocks).
        chunks: list[str] = []
        for item in getattr(resp, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    chunks.append(getattr(content, "text", "") or "")
        text = "\n".join(chunks)
    return str(text).strip(), {}


# ---------------------------------------------------------------------------
# Mode: agent — LLM with VeraGrid tools
# ---------------------------------------------------------------------------

def answer_agent(
    model: str,
    question: str,
    workspace: str,
    api_key_openai: str | None = None,
    api_key_claude: str | None = None,
    api_key_cursor: str | None = None,
    circuit_data: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call the LLM in agent mode with VeraGrid tools."""
    from agent import run_openai_agent, run_claude_agent, run_cursor_agent

    if _is_cursor_model(model):
        response = run_cursor_agent(
            model, question, workspace,
            api_key=api_key_cursor,
            circuit_data=circuit_data,
            return_trace=True,
            no_tools=False,
        )
    elif model.lower().startswith("claude"):
        response = run_claude_agent(
            model, question, workspace,
            api_key=api_key_claude,
            circuit_data=circuit_data,
            return_trace=True,
        )
    else:
        response = run_openai_agent(
            model, question, workspace,
            api_key=api_key_openai,
            circuit_data=circuit_data,
            return_trace=True,
        )
    if isinstance(response, dict):
        return str(response.get("answer", "")), dict(response.get("trace") or {})
    return str(response), {}


def normalize_difficulty_bucket(label: str) -> str:
    text = str(label or "").strip().lower()
    if text in {"easy", "medium", "hard"}:
        return text
    if "manual" in text:
        return "manual"
    return "other"


def build_difficulty_stats(results: list[EvalResult]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {
        "easy": {"total": 0, "correct": 0},
        "medium": {"total": 0, "correct": 0},
        "hard": {"total": 0, "correct": 0},
        "manual": {"total": 0, "correct": 0},
        "other": {"total": 0, "correct": 0},
    }
    for r in results:
        b = normalize_difficulty_bucket(r.difficulty)
        buckets[b]["total"] += 1
        if r.correct:
            buckets[b]["correct"] += 1
    for b in buckets.values():
        total = b["total"]
        b["accuracy_pct"] = round((b["correct"] / total * 100.0), 2) if total else 0.0
    return buckets


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate(
    model: str,
    mode: str,
    questions_file: str,
    output_file: str,
    delay: float = 0.0,
    system_prompt: str | None = None,
    api_key_openai: str | None = None,
    api_key_claude: str | None = None,
    api_key_cursor: str | None = None,
    workspace_base: str | None = None,
    circuit_data: dict[str, Any] | None = None,
) -> dict[str, Any]:

    mode = canonical_mode(mode)
    questions = load_questions(questions_file)
    results: list[EvalResult] = []
    correct = 0

    print(f"\n{'='*65}")
    print(f"  Model:     {model}")
    print(f"  Mode:      {mode}")
    print(f"  Questions: {len(questions)}")
    print(f"{'='*65}\n")

    for q in questions:
        options_text = "\n".join(f"{l}. {v}" for l, v in q.options)
        user_prompt = f"Q{q.question_id}. {q.question}\n{options_text}"

        # Create a per-question workspace for agent mode
        workspace = None
        evaluation_trace: dict[str, Any] = {
            "used_tools": False,
            "tool_call_count": 0,
            "wrote_files": False,
            "simulation_executed": False,
            "read_files": False,
            "listed_files": False,
            "tool_calls": [],
            "errors": [],
        }
        try:
            if mode == "agent":
                workspace = tempfile.mkdtemp(
                    prefix=f"vg_q{q.question_id}_",
                    dir=workspace_base,
                )
                # Seed workspace with active circuit for tools like run_web_opf.
                if circuit_data:
                    with open(os.path.join(workspace, "active_circuit_model.json"), "w", encoding="utf-8") as f:
                        json.dump(circuit_data, f, indent=2)
                raw_answer, agent_trace = answer_agent(
                    model, user_prompt, workspace,
                    api_key_openai=api_key_openai,
                    api_key_claude=api_key_claude,
                    api_key_cursor=api_key_cursor,
                    circuit_data=circuit_data,
                )
                if agent_trace:
                    evaluation_trace = agent_trace
            else:
                raw_answer, no_tool_trace = answer_no_tool(
                    model, user_prompt,
                    system_prompt=system_prompt,
                    api_key_openai=api_key_openai,
                    api_key_claude=api_key_claude,
                    api_key_cursor=api_key_cursor,
                    workspace_base=workspace_base,
                    circuit_data=circuit_data,
                )
                if no_tool_trace:
                    evaluation_trace = no_tool_trace

            model_letter = extract_letter(raw_answer)

        except Exception as exc:
            raw_answer = f"ERROR: {exc}"
            model_letter = "?"
        finally:
            # Clean up workspace
            if workspace and os.path.exists(workspace):
                shutil.rmtree(workspace, ignore_errors=True)

        is_correct = model_letter == q.expected
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"  Q{q.question_id:>3}: model={model_letter} expected={q.expected}  {status}")

        results.append(EvalResult(
            question_id=q.question_id,
            question=q.question,
            model_raw=raw_answer,
            model_answer=model_letter,
            expected=q.expected,
            correct=is_correct,
            mode=mode,
            requires_simulation=q.requires_simulation,
            difficulty=q.difficulty,
            category=q.category,
            source=q.source,
            tool_calls=int(evaluation_trace.get("tool_call_count", 0) or 0),
            evaluation_trace=evaluation_trace,
        ))

        if delay > 0:
            time.sleep(delay)

    total = len(questions)
    accuracy = (correct / total * 100) if total else 0.0

    # Breakdown by question type
    sim_qs = [r for r in results if r.requires_simulation]
    knowledge_qs = [r for r in results if not r.requires_simulation]
    sim_acc = (sum(1 for r in sim_qs if r.correct) / len(sim_qs) * 100) if sim_qs else 0.0
    know_acc = (sum(1 for r in knowledge_qs if r.correct) / len(knowledge_qs) * 100) if knowledge_qs else 0.0
    difficulty_stats = build_difficulty_stats(results)

    print(f"\n{'='*65}")
    print(f"  RESULTS: {correct}/{total} correct — {accuracy:.1f}%")
    if sim_qs:
        print(f"    Simulation Qs: {sum(1 for r in sim_qs if r.correct)}/{len(sim_qs)} — {sim_acc:.1f}%")
    if knowledge_qs:
        print(f"    Knowledge Qs:  {sum(1 for r in knowledge_qs if r.correct)}/{len(knowledge_qs)} — {know_acc:.1f}%")
    for label in ("easy", "medium", "hard", "manual"):
        d = difficulty_stats[label]
        if d["total"] > 0:
            print(f"    {label.title()} Qs:      {d['correct']}/{d['total']} — {d['accuracy_pct']:.1f}%")
    print(f"{'='*65}\n")

    output = {
        "model": model,
        "mode": "no_tool_use" if mode == "no_tool_access" else mode,
        "questions_file": questions_file,
        "total_questions": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "accuracy_simulation_pct": round(sim_acc, 2),
        "accuracy_knowledge_pct": round(know_acc, 2),
        "accuracy_by_difficulty": difficulty_stats,
        "wrong_question_ids": [r.question_id for r in results if not r.correct],
        "wrong_ids": [r.question_id for r in results if not r.correct],
        "details": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "difficulty": r.difficulty,
                "category": r.category,
                "source": r.source,
                "model_raw": r.model_raw,
                "model_answer": r.model_answer,
                "expected": r.expected,
                "correct": r.correct,
                "tool_calls": r.tool_calls,
                "requires_simulation": r.requires_simulation,
                "evaluation_trace": r.evaluation_trace,
            }
            for r in results
        ],
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to: {output_file}")

    return output


# ---------------------------------------------------------------------------
# Comparison mode: run both and produce a diff report
# ---------------------------------------------------------------------------

def compare_modes(
    model: str,
    questions_file: str,
    output_dir: str = ".",
    delay: float = 0.0,
    api_key_openai: str | None = None,
    api_key_claude: str | None = None,
    api_key_cursor: str | None = None,
    circuit_data: dict[str, Any] | None = None,
) -> None:
    """Run both modes and produce a comparison report."""

    print("\n" + "=" * 65)
    print("  COMPARISON RUN: no_tool_access vs agent")
    print("=" * 65)

    no_tool_out = os.path.join(output_dir, "results_no_tool.json")
    agent_out = os.path.join(output_dir, "results_agent.json")

    r1 = evaluate(model, "no_tool_access", questions_file, no_tool_out,
                  delay=delay, api_key_openai=api_key_openai,
                  api_key_claude=api_key_claude, api_key_cursor=api_key_cursor,
                  circuit_data=circuit_data)
    r2 = evaluate(model, "agent", questions_file, agent_out,
                  delay=delay, api_key_openai=api_key_openai,
                  api_key_claude=api_key_claude, api_key_cursor=api_key_cursor,
                  circuit_data=circuit_data)

    print("\n" + "=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Mode':<20} {'Overall':>10} {'Simulation':>12} {'Knowledge':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'no_tool_access':<20} {r1['accuracy_pct']:>9.1f}% {r1['accuracy_simulation_pct']:>11.1f}% {r1['accuracy_knowledge_pct']:>11.1f}%")
    print(f"  {'agent':<20} {r2['accuracy_pct']:>9.1f}% {r2['accuracy_simulation_pct']:>11.1f}% {r2['accuracy_knowledge_pct']:>11.1f}%")

    delta = r2["accuracy_pct"] - r1["accuracy_pct"]
    print(f"\n  Agent advantage: {delta:+.1f}%")

    # Find questions where agent got it right but no-tool didn't
    d1 = {d["question_id"]: d for d in r1["details"]}
    d2 = {d["question_id"]: d for d in r2["details"]}
    agent_wins = [qid for qid in d1 if not d1[qid]["correct"] and d2.get(qid, {}).get("correct")]
    agent_losses = [qid for qid in d1 if d1[qid]["correct"] and not d2.get(qid, {}).get("correct")]

    if agent_wins:
        print(f"  Agent solved (no-tool missed): {agent_wins}")
    if agent_losses:
        print(f"  Agent missed (no-tool solved): {agent_losses}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_circuit_file(path: str | None) -> dict[str, Any] | None:
    """Load circuit data from a JSON file (exported from frontend)."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Accept either:
    # 1) frontend model: {GENERATORS, LOADS, LINES}
    # 2) prompt-builder model: {buses, branches, generators}
    if isinstance(raw, dict) and {"GENERATORS", "LOADS", "LINES"}.issubset(raw.keys()):
        from prompt_builder import frontend_model_to_circuit_data

        return frontend_model_to_circuit_data(raw)
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VeraGrid MCQ Evaluator — No-Tool vs Agent Mode"
    )
    parser.add_argument("--model", default="gpt-4o", help="Model name (e.g. gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--mode", choices=["no_tool_access", "no_tool_use", "agent"], default="no_tool_access")
    parser.add_argument("--questions-file", default="mcq_questions.json", help="MCQ JSON file")
    parser.add_argument("--circuit-file", default=None,
                        help="JSON file with circuit data from frontend (for agent mode)")
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between API calls (seconds)")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt override")
    parser.add_argument("--compare", action="store_true", help="Run both modes and compare")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--claude-api-key", default="")
    parser.add_argument("--cursor-api-key", default="",
                        help="API key for the cursor-agent CLI (cursor:* models)")
    args = parser.parse_args()

    circuit = _load_circuit_file(args.circuit_file)

    try:
        if args.compare:
            compare_modes(
                model=args.model,
                questions_file=args.questions_file,
                delay=args.delay,
                api_key_openai=args.openai_api_key,
                api_key_claude=args.claude_api_key,
                api_key_cursor=args.cursor_api_key,
                circuit_data=circuit,
            )
        else:
            evaluate(
                model=args.model,
                mode=args.mode,
                questions_file=args.questions_file,
                output_file=args.output,
                delay=args.delay,
                system_prompt=args.system_prompt,
                api_key_openai=args.openai_api_key,
                api_key_claude=args.claude_api_key,
                api_key_cursor=args.cursor_api_key,
                circuit_data=circuit,
            )
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)