"""
scorer.py — LLM-based resume scoring using real API calls with resilient retries.
"""

import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ScoringWeights:
    """Relative importance of each scoring dimension."""
    skills: float = 0.40
    experience: float = 0.30
    education: float = 0.15
    cultural_fit: float = 0.15


@dataclass
class ScoreResult:
    """Full scoring result for a single resume."""
    filename: str = ""
    overall_score: float = 0.0
    recommendation: str = "No"
    skills_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    cultural_fit_score: float = 0.0
    years_of_experience: float = 0.0
    matched_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)
    education_level: str = ""
    summary: str = ""
    justification: str = ""
    tokens_used: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ResumeScorer:
    """Score a resume against a job description using an LLM."""

    def __init__(self, api_key: str, provider: str = "gemini", model: str = None, weights: ScoringWeights = None):
        self.api_key = api_key
        self.provider = provider
        self.model = model or self._default_model()
        self.weights = weights or ScoringWeights()

    def _default_model(self) -> str:
        defaults = {
            "gemini":    "gemini-2.5-flash", 
            "anthropic": "claude-haiku-4-5-20251001",
            "openai":    "gpt-4o-mini",
        }
        return defaults.get(self.provider, "gemini-2.5-flash")

    def _build_prompt(self, resume_text: str, job_description: str) -> str:
        return f"""You are an expert HR recruiter. Carefully evaluate the resume below against the job description.

CRITICAL RULES:
- If the candidate's domain/field does NOT match the job at all (e.g. software engineer applying for accounting), all scores must be very low (below 25).
- Be strict and realistic. Do not give high scores just because the candidate has experience — it must be RELEVANT experience.
- Skills score should reflect how many required skills from the job description the candidate actually has.
- Only return valid JSON matching the exact key structure required below.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Return ONLY this JSON structure:
{{
  "skills_score": <0-100>,
  "experience_score": <0-100>,
  "education_score": <0-100>,
  "cultural_fit_score": <0-100>,
  "years_of_experience": <number>,
  "education_level": "<highest degree>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "summary": "<2 sentence candidate summary>",
  "justification": "<2 sentence explanation of scores>"
}}"""

    def _call_gemini(self, prompt: str) -> tuple[str, int]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1
            }
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        
        max_retries = 4
        initial_backoff = 2

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
                return text, tokens

            except urllib.error.HTTPError as e:
                if e.code in [429, 503, 504]:
                    if attempt < max_retries - 1:
                        sleep_time = initial_backoff * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise e

    def _call_anthropic(self, prompt: str) -> tuple[str, int]:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["content"][0]["text"]
        tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
        return text, tokens

    def _call_openai(self, prompt: str) -> tuple[str, int]:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return text, tokens

    def _parse_json(self, text: str) -> dict:
        # Clean regex to strip markdown enclosures safely on a single line
        clean_text = re.sub(r"```json|```", "", text).strip()
        return json.loads(clean_text)

    def score(self, resume_text: str, job_description: str, filename: str = "resume") -> ScoreResult:
        """Score a single resume against the job description using the LLM with guaranteed parsing safety."""
        result = ScoreResult(filename=filename)
        prompt = self._build_prompt(resume_text, job_description)

        try:
            if self.provider == "gemini":
                raw, tokens = self._call_gemini(prompt)
            elif self.provider == "anthropic":
                raw, tokens = self._call_anthropic(prompt)
            elif self.provider == "openai":
                raw, tokens = self._call_openai(prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            data = self._parse_json(raw)
            
            result.skills_score       = float(data.get("skills_score", 0) or 0)
            result.experience_score   = float(data.get("experience_score", 0) or 0)
            result.education_score    = float(data.get("education_score", 0) or 0)
            result.cultural_fit_score = float(data.get("cultural_fit_score", 0) or 0)
            result.years_of_experience = float(data.get("years_of_experience", 0) or 0)
            result.education_level    = str(data.get("education_level", "Not Specified"))
            result.matched_skills     = list(data.get("matched_skills", []))
            result.missing_skills     = list(data.get("missing_skills", []))
            result.summary            = str(data.get("summary", "No summary provided."))
            result.justification      = str(data.get("justification", "No justification provided."))
            result.tokens_used        = tokens

        except Exception as e:
            result.error = str(e)
            result.summary = "Failed to evaluate candidate due to API connectivity limits."
            result.justification = f"Detailed error context: {str(e)}"

        # Calculate weighted overall score
        result.overall_score = round(
            result.skills_score       * self.weights.skills +
            result.experience_score   * self.weights.experience +
            result.education_score    * self.weights.education +
            result.cultural_fit_score * self.weights.cultural_fit,
            1
        )

        # Recommendation thresholds
        if result.overall_score >= 80:
            result.recommendation = "Strong Yes"
        elif result.overall_score >= 65:
            result.recommendation = "Yes"
        elif result.overall_score >= 45:
            result.recommendation = "Maybe"
        else:
            result.recommendation = "No"

        return result