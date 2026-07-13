"""
helpers/llm_client.py
=====================
Thin client for LLM-Score (taekb/llm-select) over a local Ollama model.

LLM-Score queries the model once per feature and asks for a single importance
score in [0, 1] for predicting the target. We force JSON output via Ollama's
`format: "json"` option and parse `{"score": float, "reasoning": str}`, with a
regex fallback for a bare number and a couple of retries on malformed output.

Uses `httpx` (already in the project venv) against the Ollama REST API — no
extra dependency, no `ollama` pip package required.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

SYSTEM_PROMPT = (
    "You are an expert in US domestic airline operations and flight-delay "
    "forecasting. You assess how useful individual input features are for a "
    "machine-learning model. For the feature you are given, output a single "
    "importance score between 0 and 1 for predicting {target}, where 0 means "
    "the feature is irrelevant / pure noise and 1 means it is highly predictive. "
    "Judge predictive usefulness, not merely whether the feature is related to "
    "flights. Respond with ONLY a JSON object of the form "
    '{{"score": <float 0-1>, "reasoning": "<one short sentence>"}} '
    "and nothing else."
)

_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


@dataclass
class ScoreResult:
    score: float
    reasoning: str
    raw: str          # raw model content, kept for the audit trail


def build_user_prompt(feature: str, meta: dict, data_block: str | None) -> str:
    """Assemble the per-feature user message for one variant."""
    unit = meta.get("unit")
    lines = [
        f"Feature name: {feature}",
        f"Type: {meta.get('type', 'unknown')}",
    ]
    if unit:
        lines.append(f"Unit: {unit}")
    lines.append(f"Description: {meta.get('description', '')}")
    if data_block:
        lines.append("")
        lines.append(
            "Below are 100 example flights (comma-separated, with a header). "
            "The final column DELAYED is the label (1 = delayed, 0 = on-time). "
            "Use any relationship you can see between this feature's column and "
            "DELAYED, together with your domain knowledge, to judge its "
            "predictive usefulness:"
        )
        lines.append(data_block)
    lines.append("")
    lines.append(
        f"Now output the JSON importance score for feature '{feature}'."
    )
    return "\n".join(lines)


def _parse_score(content: str) -> tuple[float | None, str]:
    """Return (score in [0,1] or None, reasoning)."""
    reasoning = ""
    # Preferred path: a JSON object somewhere in the content.
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            reasoning = str(obj.get("reasoning", "")).strip()
            if "score" in obj:
                return _clamp(float(obj["score"])), reasoning
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback: first number in the text.
    num = _NUM_RE.search(content)
    if num:
        try:
            return _clamp(float(num.group(1))), reasoning
        except ValueError:
            pass
    return None, reasoning


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class OllamaScorer:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.5,
        timeout: float = 120.0,
        target: str = "flight delay",
    ):
        self.model = model
        self.url = host.rstrip("/") + "/api/chat"
        self.temperature = temperature
        self.timeout = timeout
        self.system = SYSTEM_PROMPT.format(target=target)

    def score_feature(
        self,
        feature: str,
        meta: dict,
        data_block: str | None = None,
        seed: int | None = None,
        max_retries: int = 2,
    ) -> ScoreResult:
        user = build_user_prompt(feature, meta, data_block)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        if seed is not None:
            payload["options"]["seed"] = seed

        last_raw = ""
        for attempt in range(max_retries + 1):
            resp = httpx.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            last_raw = content
            score, reasoning = _parse_score(content)
            if score is not None:
                return ScoreResult(score=score, reasoning=reasoning, raw=content)
            # nudge temperature up slightly on retry to escape a bad completion
            payload["options"]["temperature"] = min(
                1.0, payload["options"]["temperature"] + 0.2
            )
        raise ValueError(
            f"Could not parse a score for feature '{feature}' after "
            f"{max_retries + 1} attempts. Last response: {last_raw!r}"
        )
