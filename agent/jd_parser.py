from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from agent.llm_manager import LLMManager, LLMRequestError

logger = logging.getLogger(__name__)


class ParsedJobDescription(BaseModel):
    job_title: str = "unknown"
    required_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    experience_level: str = "unknown"  # junior|mid|senior|staff|unknown
    experience_years_min: Optional[int] = None
    experience_years_max: Optional[int] = None
    location_type: str = "unknown"  # remote|hybrid|onsite|flexible|unknown
    location: Optional[str] = None
    company_culture: List[str] = Field(default_factory=list)
    salary_range: Optional[str] = None
    confidence_score: float = 0.0
    raw_jd: str

    @classmethod
    def sanitize(cls, payload: Dict[str, Any], raw_jd: str) -> "ParsedJobDescription":
        safe: Dict[str, Any] = {
            "job_title": cls._to_str(payload.get("job_title"), default="unknown"),
            "required_skills": cls._to_str_list(payload.get("required_skills")),
            "nice_to_have_skills": cls._to_str_list(payload.get("nice_to_have_skills")),
            "experience_level": cls._normalize_experience_level(payload.get("experience_level")),
            "experience_years_min": cls._to_optional_int(payload.get("experience_years_min")),
            "experience_years_max": cls._to_optional_int(payload.get("experience_years_max")),
            "location_type": cls._normalize_location_type(payload.get("location_type")),
            "location": cls._to_optional_str(payload.get("location")),
            "company_culture": cls._to_str_list(payload.get("company_culture")),
            "salary_range": cls._to_optional_str(payload.get("salary_range")),
            "confidence_score": cls._to_confidence(payload.get("confidence_score")),
            "raw_jd": raw_jd,
        }
        return cls(**safe)

    @staticmethod
    def _to_str(value: Any, default: str = "") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    @staticmethod
    def _to_optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _to_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_str_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
            return list(dict.fromkeys(items))
        if isinstance(value, str):
            items = [p.strip() for p in re.split(r"[,\n;/]", value) if p.strip()]
            return list(dict.fromkeys(items))
        return []

    @staticmethod
    def _normalize_experience_level(value: Any) -> str:
        allowed = {"junior", "mid", "senior", "staff", "unknown"}
        text = str(value).strip().lower() if value is not None else "unknown"
        if text in {"jr", "entry", "entry-level"}:
            text = "junior"
        elif text in {"middle", "mid-level"}:
            text = "mid"
        elif text in {"sr", "lead", "principal"}:
            text = "senior"
        return text if text in allowed else "unknown"

    @staticmethod
    def _normalize_location_type(value: Any) -> str:
        allowed = {"remote", "hybrid", "onsite", "flexible", "unknown"}
        text = str(value).strip().lower() if value is not None else "unknown"
        if "remote" in text:
            text = "remote"
        elif "hybrid" in text:
            text = "hybrid"
        elif text in {"on-site", "on site"}:
            text = "onsite"
        elif "onsite" in text:
            text = "onsite"
        elif "flex" in text:
            text = "flexible"
        return text if text in allowed else "unknown"

    @staticmethod
    def _to_confidence(value: Any) -> float:
        try:
            score = float(value)
            if score < 0.0:
                return 0.0
            if score > 1.0:
                return 1.0
            return score
        except (TypeError, ValueError):
            return 0.0


class JDParser:
    def __init__(self, llm_manager: LLMManager) -> None:
        self._llm_manager = llm_manager

    async def parse(self, job_description: str) -> ParsedJobDescription:
        if not job_description or not job_description.strip():
            raise ValueError("job_description cannot be empty.")

        raw_jd = job_description.strip()
        primary_prompt = self._build_prompt(raw_jd, simplified=False)
        fallback_prompt = self._build_prompt(raw_jd, simplified=True)

        parsed_payload: Dict[str, Any] = {}

        try:
            response = await self._llm_manager.generate(
                prompt=primary_prompt,
                system_prompt="You are an expert recruiter analyzing job descriptions.",
                complexity="medium",
                temperature=0.0,
                max_tokens=1500,
                metadata={"task": "jd_parse_primary"},
            )
            parsed_payload = self._safe_parse_json(response.text)
        except (LLMRequestError, json.JSONDecodeError) as exc:
            logger.warning("Primary JD parse attempt failed: %s", exc)

        if not parsed_payload:
            try:
                response = await self._llm_manager.generate(
                    prompt=fallback_prompt,
                    system_prompt="You are an expert recruiter analyzing job descriptions.",
                    complexity="medium",
                    temperature=0.0,
                    max_tokens=700,
                    metadata={"task": "jd_parse_retry"},
                )
                parsed_payload = self._safe_parse_json(response.text)
            except (LLMRequestError, json.JSONDecodeError) as exc:
                logger.warning("Retry JD parse attempt failed: %s", exc)

        if not parsed_payload:
            logger.warning("LLM returned invalid JSON twice. Falling back to heuristic parse.")
            parsed_payload = self._heuristic_fallback(raw_jd)

        result = self._build_best_effort(parsed_payload, raw_jd)
        self._log_missing_fields(result)
        return result

    def _build_prompt(self, raw_jd: str, simplified: bool) -> str:
        if simplified:
            return (
                "Return ONLY valid JSON. No text before or after.\n"
                "{\n"
                '"job_title": "extract title",\n'
                '"required_skills": ["skill1", "skill2"],\n'
                '"nice_to_have_skills": [],\n'
                '"experience_level": "junior|mid|senior|staff|unknown",\n'
                '"experience_years_min": null,\n'
                '"experience_years_max": null,\n'
                '"location_type": "remote|hybrid|onsite|flexible|unknown",\n'
                '"location": null,\n'
                '"company_culture": [],\n'
                '"salary_range": null,\n'
                '"confidence_score": 0.5\n'
                "}\n"
                "Extract from this job description:\n"
                f"{raw_jd}"
            )

        return (
            "You are a JSON extraction API. Your output MUST be valid JSON and nothing else.\n"
            "Extract these fields from the job description:\n\n"
            "job_title (string)\n"
            "required_skills (array of strings)\n"
            "nice_to_have_skills (array of strings)\n"
            'experience_level (one of: "junior", "mid", "senior", "staff", "unknown")\n'
            "experience_years_min (number or null)\n"
            "experience_years_max (number or null)\n"
            'location_type (one of: "remote", "hybrid", "onsite", "flexible", "unknown")\n'
            "location (string or null)\n"
            'company_culture (array of strings like "startup", "fast-paced")\n'
            "salary_range (string or null)\n"
            "confidence_score (number 0.0-1.0)\n\n"
            "Output ONLY the JSON object. No explanation. No markdown. "
            "Start with { and end with }.\n"
            "JOB DESCRIPTION:\n"
            f"{raw_jd}\n"
            "JSON OUTPUT:"
        )

    def _safe_parse_json(self, llm_text: str) -> Dict[str, Any]:
        if not llm_text or not llm_text.strip():
            raise json.JSONDecodeError("Empty LLM output", llm_text or "", 0)

        candidate = llm_text.strip()

        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()

        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                raise json.JSONDecodeError("Root is not object", candidate, 0)
            return payload
        except json.JSONDecodeError:
            pass

        extracted = self._extract_json_object(candidate)
        payload = json.loads(extracted)
        if not isinstance(payload, dict):
            raise json.JSONDecodeError("Root is not object", extracted, 0)
        return payload

    def _extract_json_object(self, text: str) -> str:
        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object start found", text, 0)

        depth = 0
        in_string = False
        escape = False

        for i, ch in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        raise json.JSONDecodeError("No complete JSON object found", text, start)

    def _build_best_effort(self, payload: Dict[str, Any], raw_jd: str) -> ParsedJobDescription:
        try:
            return ParsedJobDescription.sanitize(payload, raw_jd=raw_jd)
        except ValidationError as exc:
            logger.warning("Validation error while building ParsedJobDescription: %s", exc)
            return ParsedJobDescription(raw_jd=raw_jd)

    def _log_missing_fields(self, result: ParsedJobDescription) -> None:
        missing: List[str] = []
        if result.job_title == "unknown":
            missing.append("job_title")
        if not result.required_skills:
            missing.append("required_skills")
        if result.experience_level == "unknown":
            missing.append("experience_level")
        if result.location_type == "unknown":
            missing.append("location_type")
        if result.confidence_score <= 0.0:
            missing.append("confidence_score")

        if missing:
            logger.warning("JD parser missing/low-confidence fields: %s", ", ".join(missing))

    def _heuristic_fallback(self, raw_jd: str) -> Dict[str, Any]:
        text = raw_jd.lower()

        title = "unknown"
        first_line = raw_jd.splitlines()[0].strip() if raw_jd.splitlines() else ""
        if first_line and len(first_line) <= 80:
            title = first_line

        required_skills = []
        for skill in [
            "python",
            "django",
            "fastapi",
            "postgresql",
            "redis",
            "docker",
            "aws",
            "kubernetes",
            "react",
            "graphql",
            "sql",
        ]:
            if skill in text:
                required_skills.append(skill)

        exp_level = "unknown"
        if "senior" in text or "5+" in text or "sr " in text:
            exp_level = "senior"
        elif "junior" in text or "entry" in text:
            exp_level = "junior"
        elif "mid" in text:
            exp_level = "mid"
        elif "staff" in text:
            exp_level = "staff"

        years_min = None
        years_max = None
        year_match = re.search(r"(\d+)\s*\+?\s*years?", text)
        if year_match:
            years_min = int(year_match.group(1))

        location_type = "unknown"
        if "remote" in text:
            location_type = "remote"
        elif "hybrid" in text:
            location_type = "hybrid"
        elif "onsite" in text or "on-site" in text:
            location_type = "onsite"

        salary_range = None
        salary_match = re.search(r"\$[\d,]+[kK]?\s*[-–]\s*\$?[\d,]+[kK]?", raw_jd)
        if salary_match:
            salary_range = salary_match.group(0)

        culture = []
        for marker in ["startup", "fast-paced", "collaborative", "ownership", "agile"]:
            if marker in text:
                culture.append(marker)

        return {
            "job_title": title or "unknown",
            "required_skills": required_skills,
            "nice_to_have_skills": [],
            "experience_level": exp_level,
            "experience_years_min": years_min,
            "experience_years_max": years_max,
            "location_type": location_type,
            "location": None,
            "company_culture": culture,
            "salary_range": salary_range,
            "confidence_score": 0.35,
        }


async def run_jd_parser_examples(llm_manager: LLMManager) -> List[ParsedJobDescription]:
    parser = JDParser(llm_manager)

    examples = [
        """Senior Python Engineer
We're looking for a senior backend engineer with 5+ years Python experience.
Must have:

Python, Django/FastAPI
PostgreSQL, Redis
Docker, AWS
REST API design

Nice to have:

Kubernetes
React
GraphQL

Location: Remote (US timezone preferred)
Salary: $140k-180k
We're a fast-paced startup building developer tools.""",
        "Backend Developer needed. Experience with Python and databases required.",
        "lookin 4 coder!!! must know stuff. pay good. hmu",
    ]

    results: List[ParsedJobDescription] = []
    for jd in examples:
        try:
            parsed = await parser.parse(jd)
            results.append(parsed)
        except Exception as exc:
            logger.exception("Unexpected parser error in test runner: %s", exc)
            results.append(ParsedJobDescription(raw_jd=jd))
    return results
