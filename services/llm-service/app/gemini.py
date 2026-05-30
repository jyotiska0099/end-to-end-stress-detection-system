import json
import logging
import re

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


class MockUsageMetadata:
    prompt_token_count = 120
    candidates_token_count = 40


class MockResponse:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = MockUsageMetadata()


class MockGenerativeModel:
    def generate_content(self, prompt: str) -> MockResponse:
        # Extract stress probability from the prompt
        match = re.search(
            r"Stress probability:\s*([0-9]*\.?[0-9]+)",
            prompt,
            re.IGNORECASE,
        )

        stress_probability = float(match.group(1)) if match else 0.0

        if stress_probability < 0.35:
            payload = {
                "severity": "none",
                "urgency_level": 1,
                "summary": ("Patient shows no significant indicators of physiological stress."),
                "recommendation": (
                    "Continue routine monitoring and maintain standard care procedures."
                ),
                "follow_up_minutes": 60,
            }

        elif stress_probability < 0.55:
            payload = {
                "severity": "mild",
                "urgency_level": 2,
                "summary": ("Patient exhibits mild indicators of physiological stress."),
                "recommendation": (
                    "Monitor for changes in stress markers and encourage rest or stress-reduction measures."
                ),
                "follow_up_minutes": 30,
            }

        elif stress_probability < 0.75:
            payload = {
                "severity": "moderate",
                "urgency_level": 3,
                "summary": ("Patient demonstrates moderate physiological stress levels."),
                "recommendation": (
                    "Increase observation frequency and assess for contributing factors. "
                    "Consider supportive interventions if stress indicators persist."
                ),
                "follow_up_minutes": 15,
            }

        else:
            payload = {
                "severity": "high",
                "urgency_level": 5,
                "summary": (
                    "Patient is showing high physiological stress levels requiring prompt attention."
                ),
                "recommendation": (
                    "Initiate immediate clinical review and closely monitor the patient. "
                    "Evaluate for acute stressors or deterioration in condition."
                ),
                "follow_up_minutes": 5,
            }

        return MockResponse(json.dumps(payload))


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
    # Uncomment the below line when quota is upgraded for Gemini API
    # model = genai.GenerativeModel(
    #     model_name=settings.gemini_model,
    #     system_instruction=SYSTEM_PROMPT,
    # )

    model = MockGenerativeModel()

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
