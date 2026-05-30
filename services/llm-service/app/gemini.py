import json
import logging

import google.generativeai as genai

from app.config import settings
from app.schemas import StressRecommendation

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a clinical-grade stress monitoring assistant embedded in a real-time
patient health system. You receive physiological stress analysis results and
return structured recommendations for care staff.

You MUST respond with ONLY a valid JSON object — no markdown, no backticks,
no extra text. The JSON must match this exact schema:

{
  "severity": "<one of: none | mild | moderate | high>",
  "urgency_level": <integer 1–5>,
  "summary": "<one sentence describing the patient's current stress state>",
  "recommendation": "<one or two sentences of actionable advice for care staff>",
  "follow_up_minutes": <integer: suggested minutes until next check-in>
}

Guidelines:
- severity "none"     → stress_probability < 0.35
- severity "mild"     → stress_probability 0.35–0.55
- severity "moderate" → stress_probability 0.55–0.75
- severity "high"     → stress_probability > 0.75
- urgency_level 1 = routine, 5 = immediate intervention required
- follow_up_minutes: 60 for none, 30 for mild, 15 for moderate, 5 for high
- Keep language calm, clinical, and concise.
""".strip()


def _build_user_prompt(patient_id: str, stress_probability: float, label: int) -> str:
    return (
        f"Patient ID: {patient_id}\n"
        f"Stress probability: {stress_probability:.3f}\n"
        f"Predicted label: {'STRESS' if label == 1 else 'NO_STRESS'}\n"
        "Generate the structured recommendation JSON."
    )


def get_recommendation(
    patient_id: str,
    stress_probability: float,
    label: int,
) -> tuple[StressRecommendation, int, int]:
    """
    Call Gemini synchronously and parse structured JSON response.

    Returns:
        (StressRecommendation, prompt_tokens, output_tokens)
    """
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=SYSTEM_PROMPT,
    )

    prompt = _build_user_prompt(patient_id, stress_probability, label)
    logger.info("Calling Gemini for patient=%s prob=%.3f", patient_id, stress_probability)

    response = model.generate_content(prompt)

    raw_text = response.text.strip()
    logger.debug("Gemini raw response: %s", raw_text)

    # Strip accidental markdown fences
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    data = json.loads(raw_text)
    recommendation = StressRecommendation(**data)

    usage = response.usage_metadata
    return recommendation, usage.prompt_token_count, usage.candidates_token_count
