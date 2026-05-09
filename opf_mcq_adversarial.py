#!/usr/bin/env python3
"""
opf_mcq_adversarial.py
======================

GAN-like adversarial MCQ generator for AC-OPF / power-flow results.

Architecture:
  - Generator agent : Claude call that uses the deterministic templates from
                      ``opf_mcq_generator_150`` as a "seed library" and
                      synthesizes NOVEL, complex MCQs with planted "trap"
                      distractors that exploit common misconceptions.
  - Solver agent    : A separate Claude call that tries to answer each MCQ
                      WITHOUT TOOLS, using only its internal reasoning. It
                      reports a confidence (low / medium / high).
  - Arbiter         : Deterministic logic. A question is "promoted" if the
                      Solver answered incorrectly OR answered correctly with
                      low confidence. These are the genuinely-hard questions.

Outputs:
  - mcq_promoted.json : The curated set that fooled (or troubled) the Solver,
                        capped per tier at the requested target counts.
  - mcq_all.json      : Every generated question with Solver verdicts attached.
  - mcq_promoted.md   : Human-readable report of the promoted set.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python opf_mcq_adversarial.py --input opf_results.json
  python opf_mcq_adversarial.py --input opf_results.json --easy 10 --medium 10 --hard 10
  python opf_mcq_adversarial.py --input opf_results.json --rounds 3
  python opf_mcq_adversarial.py --input opf_results.json --dry-run   # no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Reuse the deterministic templates as a "seed library" the Generator learns from.
# These give the LLM concrete examples of the kinds of questions/explanations we want.
from opf_mcq_generator_150 import (
    EASY_TEMPLATES,
    MEDIUM_TEMPLATES,
    HARD_TEMPLATES,
    validate_results_schema,
    MCQ as SeedMCQ,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_GEN_MODEL = "claude-sonnet-4-5"
DEFAULT_SOLVER_MODEL = "claude-sonnet-4-5"
MAX_TOKENS_GEN = 4096
MAX_TOKENS_SOLVE = 2048
# Reasoning models burn budget on hidden thinking before any visible token, so
# they need a much larger ceiling than chat-only models.
MAX_TOKENS_GEN_REASONING = 16384
MAX_TOKENS_SOLVE_REASONING = 8192
SEED_EXAMPLES_PER_TIER = 3      # how many template examples to show the Generator
API_MAX_RETRIES = 3
API_RETRY_BASE_DELAY = 2.0      # seconds; doubled each retry


# ============================================================================
# Data models
# ============================================================================

@dataclass
class AdversarialMCQ:
    id: int
    difficulty: str
    category: str
    question: str
    options: Dict[str, str]
    correct_answer: str
    correct_value: str
    explanation: str
    trap_description: str = ""
    # Filled in by the Solver/Arbiter:
    solver_choice: Optional[str] = None
    solver_confidence: Optional[str] = None
    solver_reasoning: Optional[str] = None
    solver_correct: Optional[bool] = None
    promoted: bool = False
    promotion_reason: str = ""
    round_generated: int = 1


# ============================================================================
# Prompt construction
# ============================================================================

GENERATOR_SYSTEM = """You are an expert AC-OPF (Alternating Current Optimal Power Flow) MCQ designer
in an ADVERSARIAL setting. Your goal is to FOOL a strong LLM Solver that can read the
same dataset and reason carefully step-by-step. A trivial lookup question is a failed
question.

You will be given:
  1. A complete AC-OPF results dataset (buses, branches, generators, loads, summary).
  2. Example questions at the target difficulty (the "seed library").
  3. A target difficulty tier and the number of questions to produce.

DIFFICULTY DEFINITIONS (all tiers must include a trap distractor — the tiers
differ only in HOW MANY reasoning steps the *correct* path takes):
  - Easy   : ONE non-trivial concept. The Solver must already understand a sign
             convention, a definition, or a unit before it can pick the right
             option. Pure "what is the value of X" lookups are FORBIDDEN — every
             Easy question must have at least one distractor a careless reader
             would pick.
  - Medium : One formula application or unit conversion (per-unit, apparent power,
             power factor, R/X ratio, utilization, single-step balance) PLUS a
             trap distractor.
  - Hard   : 2+ chained steps (cross-table inference, statistics, current
             magnitude from S and V, loss share, cost share, incremental cost,
             margin to limit, etc.) AND a trap distractor that survives a
             careless first-pass attempt.

ADVERSARIAL QUALITY RULES (CRITICAL — the Solver must FAIL or be uncertain):
  - Every question MUST contain a "trap" distractor: a wrong answer that a
    well-trained model would naturally pick if it skips one verification step.
    Common high-yield traps to exploit (use a DIFFERENT trap on each question):
      * Sign convention: Pf vs Pt (Ploss = Pf + Pt, NOT Pf − Pt; Pt is usually
        negative because it flows out of the to-end).
      * Per-unit vs SI: forgetting to divide by Sbase, or dividing twice.
      * Apparent vs active: |S| = sqrt(P² + Q²), not |P| + |Q| and not P alone.
      * Angle direction: Vθ_from − Vθ_to vs Vθ_to − Vθ_from.
      * Current magnitude: |I| = |S|/|V|, not |S|·|V|, not |S|/V² .
      * Reserve: Pmax − P, not P alone, not Pmax alone, not P − Pmin.
      * Cost: total cost (Pcost) vs marginal/incremental cost
        (d(Cost)/dP = 2·Cost2·P + Cost), vs per-MW cost (Pcost/P).
      * Loading %: branch flow as % of limit, NOT raw Pf.
      * "Total losses" trap: sum |Ploss| across branches vs naive Pgen − Pload.
      * Slack vs PV bus: voltage angle reference is exactly 0 at slack.
      * Reactive sign: leading vs lagging, capacitive vs inductive Q.
  - DISTRACTOR ENGINEERING (this is the whole game):
      * Each distractor MUST be the EXACT result of applying a specific
        misconception to the same numbers. Do NOT just perturb the correct
        value by ±10%.
      * At least one distractor must be within ±5% of the correct answer in
        magnitude (so eyeballing won't separate them).
      * Avoid wrong units in distractors. Same unit and same number of decimal
        places as the correct option.
      * Do not include obviously absurd options (negative MW for a load, etc.).
  - QUESTION STEMS must be context-free: "For Line 1-2, what is the active
    power loss?" — NOT "Given Pf = 98.5 MW and Pt = -97.85 MW, compute..."
    The whole point is the Solver has to do the work itself.
  - VARY the correct answer position across A/B/C/D. Do NOT cluster on A.
  - VARY the trap type across the batch. Do NOT use the same misconception
    twice in a row.
  - The "trap_description" field is REQUIRED and must name the specific
    misconception the main distractor exploits (e.g. "Used Pf alone as the loss
    instead of Pf + Pt; Pt is negative so this gives roughly twice the loss.").
    Without this field the question is useless to the pipeline.

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown fences, no commentary outside the JSON.
Schema:
{
  "questions": [
    {
      "difficulty": "Easy" | "Medium" | "Hard",
      "category": "<short category name>",
      "question": "<context-free question stem>",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "A" | "B" | "C" | "D",
      "correct_value": "<numeric answer with unit>",
      "explanation": "<step-by-step derivation: formula, substitution, result>",
      "trap_description": "<which misconception the main distractor exploits>"
    }
  ]
}
"""

SOLVER_SYSTEM = """You are a power systems engineering student taking an exam.

You have strong theoretical knowledge but NO calculator, NO lookup tables, and NO
external tools. You must answer each MCQ using only your reasoning and the
dataset provided in the prompt.

For each question:
  1. Reason step-by-step internally about what formula applies and what the
     answer should be.
  2. Commit to ONE option (A/B/C/D).
  3. Report your confidence honestly:
       "low"    if you guessed or are unsure,
       "medium" if you reasoned through it but skipped verification,
       "high"   if the answer is unambiguous and you are certain.
  4. Give one short sentence describing your reasoning.

Be honest about uncertainty. Do NOT inflate your confidence.

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown fences.
Schema:
{
  "answers": [
    {
      "id": <int>,
      "chosen": "A" | "B" | "C" | "D",
      "confidence": "low" | "medium" | "high",
      "reasoning": "<one short sentence>"
    }
  ]
}
"""


def serialize_results(results: dict) -> str:
    """Render the OPF results as a compact human-readable block for the prompts."""
    lines = []
    lines.append(f"System base: Sbase = {results['Sbase_MVA']} MVA")
    lines.append(f"Solver: converged in {results.get('iterations', '?')} iterations\n")

    lines.append("BUSES:")
    for b in results["buses"]:
        name = b.get("name", f"Bus {b.get('index', '?')}")
        btype = b.get("type", "")
        type_str = f" ({btype})" if btype else ""
        lines.append(f"  {name}{type_str}: Vm={b['Vm']} p.u., Va={b['Va_deg']}°")
    lines.append("")

    lines.append("GENERATORS:")
    for g in results["generators"]:
        name = g.get("name", f"Generator {g.get('index', '?')}")
        lines.append(
            f"  {name}: P={g['P_MW']} MW, Q={g.get('Q_Mvar', g.get('Q', 0))} Mvar, "
            f"Pmin={g.get('Pmin', '?')}, Pmax={g.get('Pmax', '?')}, "
            f"Cost2={g.get('Cost2', 0)}, Cost={g.get('Cost', 0)}, "
            f"Pcost={g.get('Pcost', 0)}"
        )
    lines.append("")

    lines.append("LOADS:")
    for ld in results["loads"]:
        name = ld.get("name", f"Load {ld.get('index', '?')}")
        lines.append(f"  {name}: P={ld['P_MW']} MW, Q={ld['Q_Mvar']} Mvar")
    lines.append("")

    lines.append("BRANCHES:")
    for br in results["branches"]:
        name = br.get("name", f"Line {br.get('index', '?')}")
        lines.append(
            f"  {name}: from_bus={br['from_bus']}, to_bus={br['to_bus']}, "
            f"r={br['r']}, x={br['x']}, "
            f"Pf={br['Pf_MW']} MW, Qf={br['Qf_Mvar']} Mvar, "
            f"Pt={br['Pt_MW']} MW, Qt={br['Qt_Mvar']} Mvar, "
            f"Ploss={br['Ploss_MW']} MW, Qloss={br['Qloss_Mvar']} Mvar, "
            f"loading={br['loading_pct']}%"
        )
    lines.append("")

    lines.append("SUMMARY:")
    for k, v in results["summary"].items():
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def build_seed_examples(results: dict, difficulty: str, n: int, rng: random.Random) -> str:
    """Generate a handful of deterministic seed questions to prime the Generator."""
    if difficulty == "Easy":
        templates = EASY_TEMPLATES
    elif difficulty == "Medium":
        templates = MEDIUM_TEMPLATES
    else:
        templates = HARD_TEMPLATES

    counter = [0]
    pool: List[SeedMCQ] = []
    for spec in templates:
        try:
            pool.extend(spec.generator(results, rng, counter))
        except Exception:
            continue

    if not pool:
        return "(no seed examples available)"

    sample = rng.sample(pool, min(n, len(pool)))
    blocks = []
    for q in sample:
        opt_text = "  ".join(f"{k}: {v}" for k, v in q.options.items())
        blocks.append(
            f"Q: {q.question}\n"
            f"Options: {opt_text}\n"
            f"Correct: {q.correct_answer} ({q.correct_value})\n"
            f"Explanation: {q.explanation}"
        )
    return "\n\n".join(blocks)


# ============================================================================
# Robust JSON extraction
# ============================================================================

def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response.

    Handles common failure modes:
      - markdown fences (```json ... ```, ```...```)
      - leading/trailing prose around the JSON
      - trailing commentary after the closing brace
      - the model returning a bare list ``[{...}, {...}]`` instead of an object
      - truncation: trim to the last balanced ``}`` before retrying parse
    Returns None if nothing parseable is found.
    """
    if not text:
        return None

    cleaned = text.strip()
    # Strip optional opening fence (```json or ```) and matching closing fence.
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return {"questions": parsed}
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")
    candidates = sorted([s for s in (obj_start, arr_start) if s != -1])
    for start in candidates:
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return {"questions": obj}
        if isinstance(obj, dict):
            return obj

    # Final rescue: the model probably ran out of tokens mid-array. Walk back to
    # the last "},\n" or "}\n]" boundary and retry from the first `{`.
    for boundary in ("}\n]", "}\n  ]", "},\n", "}\n"):
        last = cleaned.rfind(boundary)
        if last == -1:
            continue
        truncated = cleaned[: last + 1]
        if not truncated.endswith("]") and "[" in truncated:
            truncated = truncated.rstrip(", \n") + "]"
        if obj_start != -1:
            payload = "{" + truncated[obj_start + 1:]
        else:
            payload = truncated
        try:
            obj, _ = decoder.raw_decode(payload)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                return {"questions": obj}
        except json.JSONDecodeError:
            continue
    return None


# ============================================================================
# Provider detection + multi-provider LLM client wrapper
# ============================================================================

def detect_provider(model: str) -> str:
    """Map a model ID to its provider.

    Returns "anthropic" for Claude models and "openai" for GPT/o-series models.
    Raises ValueError if the model name doesn't match any known prefix.
    """
    m = model.strip().lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    raise ValueError(
        f"Cannot infer provider from model name '{model}'. "
        "Use a Claude model (anthropic) or a GPT/o-series model (openai)."
    )


class LLMClients:
    """Lazy multi-provider client holder. Builds Anthropic / OpenAI SDK clients
    on first use of each provider, so an OpenAI-only run does not require an
    Anthropic key (and vice versa).
    """

    def __init__(self, anthropic_key: Optional[str], openai_key: Optional[str]):
        self.anthropic_key = (anthropic_key or "").strip() or None
        self.openai_key = (openai_key or "").strip() or None
        self._anthropic = None
        self._openai = None

    def anthropic(self):
        if self._anthropic is not None:
            return self._anthropic
        if not self.anthropic_key:
            raise RuntimeError(
                "Anthropic API key required for a Claude model but none was provided "
                "(set --api-key / ANTHROPIC_API_KEY)."
            )
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise RuntimeError("Install the Anthropic SDK first:  pip install anthropic") from e
        self._anthropic = _anthropic.Anthropic(api_key=self.anthropic_key)
        return self._anthropic

    def openai(self):
        if self._openai is not None:
            return self._openai
        if not self.openai_key:
            raise RuntimeError(
                "OpenAI API key required for a GPT model but none was provided "
                "(set --openai-api-key / OPENAI_API_KEY)."
            )
        try:
            import openai as _openai
        except ImportError as e:
            raise RuntimeError("Install the OpenAI SDK first:  pip install openai") from e
        self._openai = _openai.OpenAI(api_key=self.openai_key)
        return self._openai


def _call_anthropic(client, *, model: str, system: str, user: str, max_tokens: int) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def _is_openai_reasoning_model(model: str) -> bool:
    """GPT-5 family and o-series are reasoning models. They MUST go through the
    Responses API: their ``max_completion_tokens`` budget is shared between
    hidden reasoning tokens and visible output, so on Chat Completions the
    visible output often comes back empty (all budget eaten by reasoning).

    The Responses API exposes a separate ``reasoning.effort`` knob to bound
    that hidden phase, which is what we want for structured JSON tasks.
    """
    m = model.lower()
    return m.startswith(("o1", "o3", "o4")) or m.startswith("gpt-5")


def _openai_response_text(resp) -> str:
    """Pull plain text out of an openai.responses object across SDK shapes."""
    text = getattr(resp, "output_text", None)
    if text:
        return text
    chunks = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "reasoning":
            continue  # internal CoT, never includes the JSON we want
        for part in getattr(item, "content", []) or []:
            t = getattr(part, "text", None)
            if t:
                chunks.append(t)
    return "".join(chunks)


def _openai_usage_summary(resp) -> str:
    """Return a one-line usage breakdown ('prompt=… reasoning=… output=…') for
    diagnostics when a Responses call returns empty text."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return "(no usage info)"
    pt = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    ot = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    rt = None
    details = getattr(usage, "output_tokens_details", None) or getattr(usage, "completion_tokens_details", None)
    if details is not None:
        rt = getattr(details, "reasoning_tokens", None)
    return f"prompt={pt} reasoning={rt} output={ot}"


def _call_openai_responses(
    client, *, model: str, system: str, user: str, max_tokens: int, reasoning_effort: str = "minimal",
) -> str:
    """Call the Responses API. The ``input`` parameter is a free-form string
    (no system role on this endpoint), so we fold the system message into a
    leading instruction block.

    Some accounts / SDK versions reject the ``reasoning`` kwarg. We try with it
    first and fall back to a plain call if rejected.
    """
    merged = f"{system}\n\n---\n\n{user}"

    def _do_call(extra_kwargs: dict):
        return client.responses.create(
            model=model,
            input=merged,
            max_output_tokens=max_tokens,
            **extra_kwargs,
        )

    try:
        resp = _do_call({"reasoning": {"effort": reasoning_effort}})
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "reasoning" in msg or "unsupported" in msg or "unknown" in msg:
            resp = _do_call({})
        else:
            raise

    text = _openai_response_text(resp)
    if text:
        return text

    # Empty output: most likely the model burned the entire budget on hidden
    # reasoning. Surface a clear diagnostic instead of returning "" silently.
    raise RuntimeError(
        f"OpenAI Responses API returned EMPTY output for model={model} "
        f"(usage: {_openai_usage_summary(resp)}). The reasoning model spent its entire "
        f"max_output_tokens={max_tokens} budget on hidden chain-of-thought before producing "
        f"a single visible character. Try a non-reasoning model (e.g. gpt-4o, gpt-4.1, "
        f"claude-haiku-4-5) or rerun with a larger token budget."
    )


def _call_openai_chat(client, *, model: str, system: str, user: str, max_tokens: int) -> str:
    """Chat Completions for GPT-4 / 4o / 4.1 family (non-reasoning models)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001 - openai SDK exceptions vary by version
        msg = str(e).lower()
        if "max_tokens" in msg and "max_completion_tokens" in msg:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    choice = resp.choices[0]
    return choice.message.content or ""


def _call_openai(client, *, model: str, system: str, user: str, max_tokens: int) -> str:
    """OpenAI dispatch.

    - GPT-4 / 4o / 4.1 family       : Chat Completions, ``max_tokens``.
    - GPT-5 family + o-series       : Responses API, ``max_output_tokens`` +
                                      ``reasoning.effort='minimal'`` so the
                                      hidden CoT doesn't eat all the budget.
    """
    if _is_openai_reasoning_model(model):
        # Reasoning models share their token budget between hidden reasoning
        # and visible output. The defaults we pass from call_generator /
        # call_solver are sized for chat models (4k / 2k); bump them up here so
        # there's room left after reasoning.
        if max_tokens < MAX_TOKENS_GEN_REASONING:
            max_tokens = max(max_tokens * 4, MAX_TOKENS_SOLVE_REASONING)
        return _call_openai_responses(
            client, model=model, system=system, user=user, max_tokens=max_tokens,
        )
    return _call_openai_chat(
        client, model=model, system=system, user=user, max_tokens=max_tokens,
    )


def _is_transient_error(err: Exception) -> bool:
    """Decide whether an SDK error is worth retrying.

    Retry on rate-limits, server-side, and connection issues. Skip retries on
    400-class client errors (bad model name, bad parameter, auth, content
    policy) — those will never succeed on retry and only waste seconds.
    """
    msg = str(err).lower()
    code = getattr(err, "status_code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code < 600):
        return True
    if any(s in msg for s in ("rate limit", "rate_limit", "overloaded", "timeout",
                              "temporarily", "503", "504", "502", "500", "529",
                              "connection", "read timed out")):
        return True
    if isinstance(code, int) and 400 <= code < 500:
        return False
    if any(s in msg for s in ("invalid_request_error", "unsupported_parameter",
                              "model_not_found", "permission_denied",
                              "authentication", "invalid api key", "401", "403",
                              "404", "400")):
        return False
    return True  # default optimistic for unknown shapes


def call_llm(clients: "LLMClients", *, model: str, system: str, user: str, max_tokens: int) -> str:
    """Dispatch to the right provider based on the model name. Retries on
    transient errors only; 400-class errors fail fast with a useful message."""
    provider = detect_provider(model)
    last_err: Optional[Exception] = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            if provider == "anthropic":
                return _call_anthropic(
                    clients.anthropic(), model=model, system=system, user=user, max_tokens=max_tokens,
                )
            return _call_openai(
                clients.openai(), model=model, system=system, user=user, max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001 - SDK exception hierarchies vary by version
            last_err = e
            if not _is_transient_error(e):
                print(
                    f"[ERROR] {provider} API rejected the request on model={model} "
                    f"(non-transient, no retry): {e}",
                    file=sys.stderr,
                )
                break
            if attempt == API_MAX_RETRIES:
                break
            delay = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"[WARN] {provider} API call failed (attempt {attempt}/{API_MAX_RETRIES}) "
                f"on model={model}: {e}. Retrying in {delay:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"{provider} API failed on model={model}: {last_err}"
    )


# ============================================================================
# Generator agent
# ============================================================================

def call_generator(
    clients: "LLMClients",
    results: dict,
    difficulty: str,
    count: int,
    rng: random.Random,
    extra_guidance: str = "",
    *,
    model: str = DEFAULT_GEN_MODEL,
) -> List[dict]:
    """Ask the chosen LLM to synthesize ``count`` MCQs at the given difficulty."""
    seed_block = build_seed_examples(results, difficulty, SEED_EXAMPLES_PER_TIER, rng)
    data_block = serialize_results(results)

    user_msg = f"""=== AC-OPF DATASET ===
{data_block}

=== SEED EXAMPLES ({difficulty}) ===
These are reference examples of the difficulty and style. Do NOT copy them
verbatim - produce NEW questions covering different elements/quantities.

{seed_block}

=== TASK ===
Generate exactly {count} {difficulty}-tier MCQ questions from the dataset above.
{extra_guidance}

Return only the JSON object."""

    text = call_llm(
        clients,
        model=model,
        system=GENERATOR_SYSTEM,
        user=user_msg,
        max_tokens=MAX_TOKENS_GEN,
    )

    parsed = extract_json(text)
    if not parsed:
        print(
            f"[ERROR] Generator ({model}) returned UNPARSEABLE output for {difficulty}-tier.\n"
            f"        Output length: {len(text)} chars. First 800 chars:\n{text[:800]}",
            file=sys.stderr,
        )
        return []

    questions = parsed.get("questions", [])
    if not isinstance(questions, list):
        print(
            f"[ERROR] Generator ({model}) JSON has no `questions` array for {difficulty}-tier. "
            f"Top-level keys: {list(parsed.keys())[:8]}",
            file=sys.stderr,
        )
        return []
    if not questions:
        print(
            f"[WARN] Generator ({model}) returned an empty `questions` array for {difficulty}-tier.",
            file=sys.stderr,
        )
    return questions


# ============================================================================
# Solver agent
# ============================================================================

def call_solver(
    clients: "LLMClients",
    results: dict,
    questions: List[AdversarialMCQ],
    *,
    model: str = DEFAULT_SOLVER_MODEL,
) -> Dict[int, dict]:
    """Ask the chosen LLM to answer each MCQ blind. Returns ``{qid: answer_dict}``."""
    if not questions:
        return {}

    data_block = serialize_results(results)
    q_block_lines = []
    for q in questions:
        opt_text = "\n    ".join(f"{k}: {v}" for k, v in q.options.items())
        q_block_lines.append(
            f"Q{q.id} [{q.difficulty}] {q.question}\n    {opt_text}"
        )
    q_block = "\n\n".join(q_block_lines)

    user_msg = f"""=== AC-OPF DATASET ===
{data_block}

=== QUESTIONS ===
Answer each one. Reason carefully but commit to one option per question.

{q_block}

Return only the JSON object."""

    text = call_llm(
        clients,
        model=model,
        system=SOLVER_SYSTEM,
        user=user_msg,
        max_tokens=MAX_TOKENS_SOLVE,
    )

    parsed = extract_json(text)
    if not parsed:
        print(
            f"[ERROR] Solver ({model}) returned UNPARSEABLE output.\n"
            f"        Output length: {len(text)} chars. First 800 chars:\n{text[:800]}",
            file=sys.stderr,
        )
        return {}

    answers = parsed.get("answers")
    if not isinstance(answers, list):
        print(
            f"[ERROR] Solver ({model}) JSON has no `answers` array. "
            f"Top-level keys: {list(parsed.keys())[:8]}",
            file=sys.stderr,
        )
        return {}

    out: Dict[int, dict] = {}
    for a in answers:
        try:
            out[int(a["id"])] = a
        except (KeyError, ValueError, TypeError):
            continue
    return out


# ============================================================================
# Arbiter
# ============================================================================

def arbitrate(
    questions: List[AdversarialMCQ],
    solver_answers: Dict[int, dict],
) -> List[AdversarialMCQ]:
    """Attach Solver verdicts and decide which questions to promote.

    Promotion rules:
      - Solver chose the wrong option           -> PROMOTED (genuinely hard)
      - Solver chose right but with low conf.   -> PROMOTED (uncertain win)
      - Solver chose right with medium/high     -> NOT promoted (too easy)
      - Solver did not answer                   -> PROMOTED (something broke,
                                                   include for manual review)
    """
    for q in questions:
        ans = solver_answers.get(q.id)
        if ans is None:
            q.promoted = True
            q.promotion_reason = "solver_no_response"
            continue

        chosen = str(ans.get("chosen", "")).strip().upper()
        confidence = str(ans.get("confidence", "medium")).strip().lower()
        reasoning = str(ans.get("reasoning", "")).strip()

        q.solver_choice = chosen
        q.solver_confidence = confidence
        q.solver_reasoning = reasoning
        q.solver_correct = (chosen == q.correct_answer)

        if not q.solver_correct:
            q.promoted = True
            q.promotion_reason = "solver_wrong"
        elif confidence == "low":
            q.promoted = True
            q.promotion_reason = "solver_low_confidence"
        else:
            q.promoted = False
            q.promotion_reason = "solver_easily_correct"

    return questions


# ============================================================================
# Pipeline orchestration
# ============================================================================

def shuffle_options(mcq: AdversarialMCQ, rng: random.Random) -> AdversarialMCQ:
    """Re-letter the options into a random A/B/C/D order so the Solver can't
    exploit any positional bias the Generator may have. Updates correct_answer
    accordingly. Mutates and returns the same object.
    """
    items = list(mcq.options.items())
    if len(items) < 2:
        return mcq

    correct_text = mcq.options.get(mcq.correct_answer)
    if correct_text is None:
        return mcq

    rng.shuffle(items)
    labels = ["A", "B", "C", "D", "E", "F"][: len(items)]
    new_options: Dict[str, str] = {}
    new_correct = mcq.correct_answer
    for label, (_, text) in zip(labels, items):
        new_options[label] = text
        if text == correct_text and new_correct == mcq.correct_answer:
            new_correct = label
    mcq.options = new_options
    mcq.correct_answer = new_correct
    return mcq


def parse_generator_output(
    raw_questions: List[dict],
    starting_id: int,
    round_num: int,
    rng: random.Random,
) -> List[AdversarialMCQ]:
    """Convert raw LLM JSON into AdversarialMCQ objects, validating each."""
    out: List[AdversarialMCQ] = []
    next_id = starting_id
    skipped_no_options = 0
    skipped_no_correct = 0
    skipped_other = 0
    for raw in raw_questions:
        try:
            opts = raw.get("options")
            if not isinstance(opts, dict) or len(opts) < 2:
                skipped_no_options += 1
                continue
            ca = str(raw.get("correct_answer", "")).strip().upper()
            normalized_opts = {str(k).strip().upper(): str(v).strip() for k, v in opts.items()}
            if ca not in normalized_opts:
                skipped_no_correct += 1
                continue
            mcq = AdversarialMCQ(
                id=next_id,
                difficulty=str(raw.get("difficulty", "Medium")).strip().capitalize(),
                category=str(raw.get("category", "General")).strip(),
                question=str(raw["question"]).strip(),
                options=normalized_opts,
                correct_answer=ca,
                correct_value=str(raw.get("correct_value", normalized_opts.get(ca, ""))).strip(),
                explanation=str(raw.get("explanation", "")).strip(),
                trap_description=str(raw.get("trap_description", "")).strip(),
                round_generated=round_num,
            )
            shuffle_options(mcq, rng)
            out.append(mcq)
            next_id += 1
        except (KeyError, TypeError, ValueError) as e:
            skipped_other += 1
            print(f"[WARN] Skipping malformed question: {e}", file=sys.stderr)
            continue

    if raw_questions and not out:
        print(
            f"[ERROR] Parsed 0/{len(raw_questions)} generator questions. "
            f"Reasons: missing/short options={skipped_no_options}, "
            f"correct_answer not in options={skipped_no_correct}, "
            f"other={skipped_other}",
            file=sys.stderr,
        )
    elif raw_questions and (skipped_no_options or skipped_no_correct or skipped_other):
        print(
            f"[INFO] Parsed {len(out)}/{len(raw_questions)} generator questions "
            f"(dropped {skipped_no_options + skipped_no_correct + skipped_other}: "
            f"options={skipped_no_options}, correct={skipped_no_correct}, other={skipped_other}).",
            file=sys.stderr,
        )
    return out


def run_round(
    clients: "LLMClients",
    results: dict,
    targets: Dict[str, int],
    round_num: int,
    starting_id: int,
    rng: random.Random,
    feedback: str = "",
    *,
    gen_model: str = DEFAULT_GEN_MODEL,
    solver_model: str = DEFAULT_SOLVER_MODEL,
) -> Tuple[List[AdversarialMCQ], int]:
    """One full Gen -> Solve -> Arbitrate cycle for all difficulty tiers.

    Returns (questions, next_id_to_use_after_this_round).
    """
    print(f"\n{'='*70}")
    print(f"ROUND {round_num}  (gen={gen_model}, solver={solver_model})")
    print(f"{'='*70}")

    all_judged: List[AdversarialMCQ] = []
    next_id = starting_id

    for difficulty in ("Easy", "Medium", "Hard"):
        target = targets.get(difficulty, 0)
        if target <= 0:
            continue

        # Generator overproduces by 50% to give the Arbiter more to choose from
        ask_for = max(target, int(target * 1.5))
        print(f"\n[Generator] {difficulty}: requesting {ask_for} questions...")
        t0 = time.time()
        raw = call_generator(clients, results, difficulty, ask_for, rng, feedback, model=gen_model)
        print(f"[Generator] {difficulty}: got {len(raw)} raw questions in {time.time()-t0:.1f}s")

        mcqs = parse_generator_output(raw, next_id, round_num, rng)
        next_id += len(mcqs)
        if not mcqs:
            print(f"[Generator] {difficulty}: no valid questions parsed - skipping tier")
            continue

        print(f"[Solver] {difficulty}: answering {len(mcqs)} questions blind...")
        t0 = time.time()
        answers = call_solver(clients, results, mcqs, model=solver_model)
        print(f"[Solver] {difficulty}: got {len(answers)} answers in {time.time()-t0:.1f}s")

        judged = arbitrate(mcqs, answers)

        n_correct = sum(1 for q in judged if q.solver_correct)
        n_promoted = sum(1 for q in judged if q.promoted)
        rate = (n_correct / len(judged) * 100) if judged else 0
        print(
            f"[Arbiter] {difficulty}: solver scored {n_correct}/{len(judged)} "
            f"({rate:.0f}%) -> promoted {n_promoted} hard questions"
        )

        all_judged.extend(judged)

    return all_judged, next_id


def build_feedback(judged: List[AdversarialMCQ]) -> str:
    """Give the next-round Generator hints about what worked."""
    if not judged:
        return ""
    fooled = [q for q in judged if q.promoted and q.solver_correct is False]
    if not fooled:
        return (
            "FEEDBACK FROM PREVIOUS ROUND: the Solver answered everything correctly. "
            "Increase the trap subtlety significantly: use multi-step questions, "
            "stack two misconceptions in one stem, and pick distractors within 5% "
            "of the correct value."
        )

    sample = fooled[: min(5, len(fooled))]
    hints = []
    for q in sample:
        if q.trap_description:
            stem = q.question if len(q.question) <= 100 else q.question[:97] + "..."
            hints.append(f"  - '{q.trap_description}'  (fooled Solver on: {stem})")
    if not hints:
        return ""
    return (
        "FEEDBACK FROM PREVIOUS ROUND - these trap types successfully fooled the Solver:\n"
        + "\n".join(hints)
        + "\nCraft new questions that exploit similar misconceptions but on different "
          "elements (different lines / generators / buses)."
    )


def run_pipeline(
    results: dict,
    targets: Dict[str, int],
    rounds: int,
    seed: int,
    *,
    gen_model: str = DEFAULT_GEN_MODEL,
    solver_model: str = DEFAULT_SOLVER_MODEL,
    anthropic_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> List[AdversarialMCQ]:
    """Full multi-round adversarial pipeline (Anthropic + OpenAI capable)."""
    anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    providers_needed = {detect_provider(gen_model), detect_provider(solver_model)}
    if "anthropic" in providers_needed and not anthropic_api_key:
        print("[FATAL] Anthropic key required for the chosen model(s) (pass --api-key or set ANTHROPIC_API_KEY)",
              file=sys.stderr)
        sys.exit(1)
    if "openai" in providers_needed and not openai_api_key:
        print("[FATAL] OpenAI key required for the chosen model(s) (pass --openai-api-key or set OPENAI_API_KEY)",
              file=sys.stderr)
        sys.exit(1)

    clients = LLMClients(anthropic_key=anthropic_api_key, openai_key=openai_api_key)
    rng = random.Random(seed)

    all_questions: List[AdversarialMCQ] = []
    feedback = ""
    next_id = 1

    for r in range(1, rounds + 1):
        promoted_so_far = {d: 0 for d in ("Easy", "Medium", "Hard")}
        for q in all_questions:
            if q.promoted and q.difficulty in promoted_so_far:
                promoted_so_far[q.difficulty] += 1

        round_targets = {
            d: max(0, targets[d] - promoted_so_far[d]) for d in ("Easy", "Medium", "Hard")
        }
        if all(v == 0 for v in round_targets.values()):
            print(f"\n[Pipeline] All tier targets met after round {r-1}. Stopping early.")
            break

        judged, next_id = run_round(
            clients,
            results,
            round_targets,
            r,
            next_id,
            rng,
            feedback,
            gen_model=gen_model,
            solver_model=solver_model,
        )
        all_questions.extend(judged)
        feedback = build_feedback(judged)

    return all_questions


# ============================================================================
# Dry-run path: synthesize fake Generator output + simulated Solver verdicts.
# Useful for testing the parsing/arbitrer pipeline without burning API tokens.
# ============================================================================

def dry_run_pipeline(
    results: dict,
    targets: Dict[str, int],
    seed: int,
) -> List[AdversarialMCQ]:
    """Bypass the API: use deterministic templates as fake "generated" questions
    and simulate a Solver that gets ~70% right with mixed confidence."""
    print("[DRY-RUN] No API calls will be made.")
    rng = random.Random(seed)
    counter = [0]

    tier_templates = {
        "Easy": EASY_TEMPLATES,
        "Medium": MEDIUM_TEMPLATES,
        "Hard": HARD_TEMPLATES,
    }
    pool: List[SeedMCQ] = []
    for tier in ("Easy", "Medium", "Hard"):
        for spec in tier_templates[tier]:
            try:
                pool.extend(spec.generator(results, rng, counter))
            except Exception:  # noqa: BLE001
                continue

    next_id = 1
    out: List[AdversarialMCQ] = []
    for tier in ("Easy", "Medium", "Hard"):
        target = targets.get(tier, 0)
        if target <= 0:
            continue
        candidates = [q for q in pool if q.difficulty == tier]
        rng.shuffle(candidates)
        ask_for = max(target, int(target * 1.5))
        chosen = candidates[:ask_for]

        for seed_q in chosen:
            mcq = AdversarialMCQ(
                id=next_id,
                difficulty=seed_q.difficulty,
                category=seed_q.category,
                question=seed_q.question,
                options=dict(seed_q.options),
                correct_answer=seed_q.correct_answer,
                correct_value=seed_q.correct_value,
                explanation=seed_q.explanation,
                trap_description="(dry-run synthetic)",
                round_generated=1,
            )
            shuffle_options(mcq, rng)
            next_id += 1
            out.append(mcq)

    fake_answers: Dict[int, dict] = {}
    for q in out:
        roll = rng.random()
        if roll < 0.7:
            chosen = q.correct_answer
            confidence = rng.choice(["medium", "high", "high"])
        else:
            wrongs = [k for k in q.options if k != q.correct_answer]
            chosen = rng.choice(wrongs) if wrongs else q.correct_answer
            confidence = rng.choice(["low", "medium"])
        fake_answers[q.id] = {
            "id": q.id,
            "chosen": chosen,
            "confidence": confidence,
            "reasoning": "(dry-run simulated reasoning)",
        }

    arbitrate(out, fake_answers)
    n_correct = sum(1 for q in out if q.solver_correct)
    n_promoted = sum(1 for q in out if q.promoted)
    print(
        f"[DRY-RUN] Generated={len(out)}  Solver-correct={n_correct}  "
        f"Promoted={n_promoted}"
    )
    return out


# ============================================================================
# Output
# ============================================================================

def _cap_promoted(promoted: List[AdversarialMCQ], targets: Dict[str, int]) -> List[AdversarialMCQ]:
    """Take up to ``targets[d]`` promoted questions per difficulty, in
    generation order (later rounds typically have better traps)."""
    capped: List[AdversarialMCQ] = []
    for diff in ("Easy", "Medium", "Hard"):
        tier = [q for q in promoted if q.difficulty == diff]
        capped.extend(tier[: targets.get(diff, 0)])
    return capped


def _build_summary(
    all_questions: List[AdversarialMCQ],
    capped: List[AdversarialMCQ],
    targets: Dict[str, int],
    seed: int,
    rounds: int,
) -> dict:
    promoted = [q for q in all_questions if q.promoted]
    by_tier = {}
    for d in ("Easy", "Medium", "Hard"):
        gen = [q for q in all_questions if q.difficulty == d]
        prom = [q for q in promoted if q.difficulty == d]
        wrong = [q for q in gen if q.solver_correct is False]
        fool_rate = (len(wrong) / len(gen) * 100) if gen else 0.0
        by_tier[d] = {
            "generated": len(gen),
            "promoted": len(prom),
            "delivered": sum(1 for q in capped if q.difficulty == d),
            "target": targets.get(d, 0),
            "solver_fool_rate_pct": round(fool_rate, 1),
        }

    return {
        "seed": seed,
        "rounds_run": rounds,
        "total_generated": len(all_questions),
        "total_promoted": len(promoted),
        "capped_promoted": len(capped),
        "by_tier": by_tier,
        "solver_score_overall": {
            "correct": sum(1 for q in all_questions if q.solver_correct),
            "wrong": sum(1 for q in all_questions if q.solver_correct is False),
            "no_answer": sum(1 for q in all_questions if q.solver_correct is None),
        },
    }


def _render_markdown(summary: dict, capped: List[AdversarialMCQ]) -> str:
    lines: List[str] = []
    lines.append("# Adversarial AC-OPF MCQ Set (Promoted Questions)")
    lines.append("")
    lines.append(
        f"- Seed: `{summary['seed']}`  |  Rounds: `{summary['rounds_run']}`  |  "
        f"Generated: `{summary['total_generated']}`  |  "
        f"Promoted: `{summary['total_promoted']}`  |  "
        f"Delivered: `{summary['capped_promoted']}`"
    )
    lines.append("")
    lines.append("## Per-tier statistics")
    lines.append("")
    lines.append("| Tier | Generated | Promoted | Delivered | Target | Solver fool-rate |")
    lines.append("|------|-----------|----------|-----------|--------|-------------------|")
    for d in ("Easy", "Medium", "Hard"):
        b = summary["by_tier"][d]
        lines.append(
            f"| {d} | {b['generated']} | {b['promoted']} | {b['delivered']} | "
            f"{b['target']} | {b['solver_fool_rate_pct']}% |"
        )
    lines.append("")

    for d in ("Easy", "Medium", "Hard"):
        tier_qs = [q for q in capped if q.difficulty == d]
        if not tier_qs:
            continue
        lines.append(f"## {d} ({len(tier_qs)} questions)")
        lines.append("")
        for i, q in enumerate(tier_qs, 1):
            lines.append(f"### {d} #{i} — {q.category}")
            lines.append("")
            lines.append(f"**Q{q.id}.** {q.question}")
            lines.append("")
            for k, v in q.options.items():
                marker = " ✅" if k == q.correct_answer else ""
                lines.append(f"- **{k}.** {v}{marker}")
            lines.append("")
            lines.append(f"**Correct:** {q.correct_answer} — `{q.correct_value}`")
            lines.append("")
            if q.explanation:
                lines.append(f"**Explanation:** {q.explanation}")
                lines.append("")
            if q.trap_description:
                lines.append(f"**Trap exploited:** {q.trap_description}")
                lines.append("")
            if q.solver_choice is not None:
                verdict = "correct" if q.solver_correct else "WRONG"
                lines.append(
                    f"**Solver:** chose `{q.solver_choice}` "
                    f"(confidence={q.solver_confidence}) — {verdict}"
                )
                if q.solver_reasoning:
                    lines.append(f"**Solver reasoning:** {q.solver_reasoning}")
                lines.append("")
            lines.append(f"_Promotion reason: {q.promotion_reason} | round {q.round_generated}_")
            lines.append("")
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


def _build_web_payload(
    capped: List[AdversarialMCQ],
    summary: dict,
    results: dict,
) -> dict:
    """Reshape the promoted set into the same JSON shape that the website's
    MCQPage / opf_mcq_generator_150 produces, so the front-end can render it
    without any special-casing.
    """
    questions = []
    for q in capped:
        questions.append({
            "id": q.id,
            "question": q.question,
            "options": q.options,
            "correct_answer": q.correct_answer,
            "correct_value": q.correct_value,
            "category": q.category,
            "difficulty": q.difficulty,
            "explanation": q.explanation,
            "source": "adversarial",
            "template_name": "adversarial",
            "trap_description": q.trap_description,
            "solver_choice": q.solver_choice,
            "solver_confidence": q.solver_confidence,
            "solver_correct": q.solver_correct,
            "promotion_reason": q.promotion_reason,
            "round_generated": q.round_generated,
        })

    by_tier = summary.get("by_tier", {})
    metadata = {
        "easy_count": by_tier.get("Easy", {}).get("delivered", 0),
        "medium_count": by_tier.get("Medium", {}).get("delivered", 0),
        "hard_count": by_tier.get("Hard", {}).get("delivered", 0),
        "total_questions": len(questions),
        "categories": sorted({q["category"] for q in questions if q.get("category")}),
        "mode": "adversarial",
        "rounds_run": summary.get("rounds_run"),
        "solver_score_overall": summary.get("solver_score_overall"),
        "solver_fool_rate_pct": {
            d: by_tier.get(d, {}).get("solver_fool_rate_pct", 0.0)
            for d in ("Easy", "Medium", "Hard")
        },
    }

    return {
        "metadata": metadata,
        "questions": questions,
        "opf_summary": results.get("summary", {}),
    }


def write_outputs(
    all_questions: List[AdversarialMCQ],
    targets: Dict[str, int],
    seed: int,
    rounds: int,
    out_all: str,
    out_promoted: str,
    out_md: Optional[str] = None,
    out_web: Optional[str] = None,
    results_for_web: Optional[dict] = None,
) -> None:
    promoted = [q for q in all_questions if q.promoted]
    capped = _cap_promoted(promoted, targets)
    summary = _build_summary(all_questions, capped, targets, seed, rounds)

    with open(out_all, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "questions": [asdict(q) for q in all_questions]},
            f,
            indent=2,
        )

    with open(out_promoted, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "questions": [asdict(q) for q in capped]},
            f,
            indent=2,
        )

    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(_render_markdown(summary, capped))

    if out_web:
        payload = _build_web_payload(capped, summary, results_for_web or {})
        with open(out_web, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total generated   : {summary['total_generated']}")
    print(f"Total promoted    : {summary['total_promoted']}")
    print(f"Delivered (capped): {summary['capped_promoted']}")
    print()
    for d in ("Easy", "Medium", "Hard"):
        b = summary["by_tier"][d]
        print(
            f"  {d:6s}: generated={b['generated']:3d}  "
            f"promoted={b['promoted']:3d}  "
            f"delivered={b['delivered']:3d} / target={b['target']:3d}  "
            f"fool-rate={b['solver_fool_rate_pct']:5.1f}%"
        )
    print()
    print(f"Solver overall: {summary['solver_score_overall']}")
    print(f"\nWrote: {out_all}")
    print(f"Wrote: {out_promoted}")
    if out_md:
        print(f"Wrote: {out_md}")
    if out_web:
        print(f"Wrote: {out_web}  (web shape)")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="GAN-like adversarial AC-OPF MCQ generator (Generator + Solver + Arbiter)"
    )
    ap.add_argument("--input", required=True, help="Path to opf_results.json")
    ap.add_argument("--out-all", default="mcq_all.json",
                    help="Output path for ALL generated questions with verdicts")
    ap.add_argument("--out-promoted", default="mcq_promoted.json",
                    help="Output path for the curated set that fooled the Solver")
    ap.add_argument("--out-md", default="mcq_promoted.md",
                    help="Output path for the markdown report (set to '' to skip)")
    ap.add_argument("--out-web", default="",
                    help="Optional path to write the promoted set in MCQPage's expected shape "
                         "(metadata + questions). Used by the website's /api/generate-mcq.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--easy", type=int, default=10, help="Target promoted Easy questions")
    ap.add_argument("--medium", type=int, default=10, help="Target promoted Medium questions")
    ap.add_argument("--hard", type=int, default=10, help="Target promoted Hard questions")
    ap.add_argument("--rounds", type=int, default=2,
                    help="Max adversarial rounds (each refines based on what fooled the Solver)")
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL,
                    help="Model ID used for the Generator agent (Claude or GPT)")
    ap.add_argument("--solver-model", default=DEFAULT_SOLVER_MODEL,
                    help="Model ID used for the Solver agent (Claude or GPT)")
    ap.add_argument("--api-key", default="",
                    help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    ap.add_argument("--openai-api-key", default="",
                    help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip API calls; use deterministic templates + simulated Solver")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    validate_results_schema(results)

    targets = {"Easy": args.easy, "Medium": args.medium, "Hard": args.hard}

    try:
        gen_provider = detect_provider(args.gen_model)
        solver_provider = detect_provider(args.solver_model)
    except ValueError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2

    print(
        f"[INFO] Adversarial MCQ pipeline starting:\n"
        f"        gen-model    = {args.gen_model}  (provider: {gen_provider})\n"
        f"        solver-model = {args.solver_model}  (provider: {solver_provider})\n"
        f"        rounds       = {args.rounds}\n"
        f"        targets      = Easy:{args.easy} Medium:{args.medium} Hard:{args.hard}\n"
        f"        dry-run      = {args.dry_run}",
        file=sys.stderr,
    )

    if args.dry_run:
        all_questions = dry_run_pipeline(results=results, targets=targets, seed=args.seed)
        rounds_run = 1
    else:
        all_questions = run_pipeline(
            results=results,
            targets=targets,
            rounds=args.rounds,
            seed=args.seed,
            gen_model=args.gen_model,
            solver_model=args.solver_model,
            anthropic_api_key=args.api_key or None,
            openai_api_key=args.openai_api_key or None,
        )
        rounds_run = args.rounds

    write_outputs(
        all_questions=all_questions,
        targets=targets,
        seed=args.seed,
        rounds=rounds_run,
        out_all=args.out_all,
        out_promoted=args.out_promoted,
        out_md=args.out_md or None,
        out_web=args.out_web or None,
        results_for_web=results,
    )

    promoted_count = sum(1 for q in all_questions if q.promoted)
    if not all_questions:
        print(
            "[FATAL] Pipeline produced 0 questions across all rounds. "
            "Most common cause: the Generator's JSON output failed to parse. "
            "Check the [ERROR] lines above for raw model output snippets.",
            file=sys.stderr,
        )
        return 2
    if promoted_count == 0:
        print(
            "[WARN] Pipeline produced questions but the Solver answered all of them correctly "
            "with non-low confidence, so 0 were Promoted. Try a weaker Solver model "
            "(e.g. claude-haiku-4-5), more rounds, or harder tier targets.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
