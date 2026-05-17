import structlog
from huggingface_hub import InferenceClient

from src.agents.state import PipelineState
from src.config import get_settings
from src.models.pipeline import ResearchBrief

log = structlog.get_logger()

LLM_PROMPT = """You are a senior equity research analyst. Based on the following extracted data,
produce a concise research brief.

Company: {ticker}
Document Type: {doc_type}
Sentiment: {sentiment} (confidence: {sentiment_confidence:.0%})

Key Financial Metrics:
{metrics}

Forward-Looking Claims (verified status):
{claims}

Named Entities Found:
{entities}

Write a structured brief with:
1. A one-line title
2. Executive summary (2-3 sentences)
3. Key findings (bullet points)
4. Risk factors (bullet points)
5. Financial highlights

Be factual. Cite the verification status of claims. Flag any unverified claims."""


class SynthesizerAgent:
    def run(self, state: PipelineState) -> PipelineState:
        if not state.extraction:
            state.errors.append("no extraction data for synthesis")
            return state

        brief = self._try_llm_synthesis(state)
        if brief is None:
            brief = self._template_synthesis(state)

        state.brief = brief
        log.info("synthesized brief", title=brief.title[:80])
        return state

    def _try_llm_synthesis(self, state: PipelineState) -> ResearchBrief | None:
        settings = get_settings()
        if not settings.hf_api_token:
            return None

        try:
            client = InferenceClient(token=settings.hf_api_token)
            prompt = self._build_prompt(state)
            response = client.chat_completion(
                model="HuggingFaceH4/zephyr-7b-beta",
                messages=[
                    {"role": "system", "content": "You are a senior equity research analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            text = response.choices[0].message.content or ""
            return self._parse_llm_brief(text, state)
        except Exception as e:
            log.warning("LLM synthesis unavailable, using template", reason=str(e)[:120])
            return None

    def _template_synthesis(self, state: PipelineState) -> ResearchBrief:
        ext = state.extraction
        if not ext:
            return ResearchBrief(company_ticker=state.company_ticker)

        ticker = state.company_ticker or "UNKNOWN"
        title = f"{ticker} Equity Research Brief — {state.doc_type.title()}"

        findings = []
        for k, v in ext.key_metrics.items():
            findings.append(f"{k.replace('_', ' ').title()}: {v}")
        for ent in ext.entities[:5]:
            findings.append(f"Entity detected: {ent.text} ({ent.label}, {ent.confidence:.0%})")

        risks = []
        for v in state.verifications:
            status = "Verified" if v.verified else "Unverified"
            risks.append(f"[{status}] {v.claim[:120]}")
        if not risks:
            risks.append("No forward-looking claims detected for verification.")

        summary = (
            f"Analysis of {ticker} {state.doc_type} filing. "
            f"Detected sentiment: {state.sentiment} "
            f"({state.sentiment_confidence:.0%} confidence). "
            f"Extracted {len(ext.key_metrics)} financial metrics "
            f"and {len(ext.entities)} named entities."
        )

        return ResearchBrief(
            company_ticker=ticker,
            title=title,
            summary=summary,
            key_findings=findings,
            risk_factors=risks,
            financial_highlights=ext.key_metrics,
            citations=[
                {"claim": v.claim[:100], "verified": str(v.verified)}
                for v in state.verifications[:10]
            ],
            confidence_score=state.sentiment_confidence,
        )

    def _build_prompt(self, state: PipelineState) -> str:
        ext = state.extraction
        if not ext:
            return ""

        metrics_str = "\n".join(f"  - {k}: {v}" for k, v in ext.key_metrics.items()) or "  None"

        claims_lines = []
        for i, claim in enumerate(ext.claims):
            ver = state.verifications[i] if i < len(state.verifications) else None
            status = "VERIFIED" if ver and ver.verified else "UNVERIFIED"
            claims_lines.append(f"  - [{status}] {claim[:150]}")
        claims_str = "\n".join(claims_lines) or "  None"

        entities_str = ", ".join(f"{e.text} ({e.label})" for e in ext.entities[:15]) or "None"

        return LLM_PROMPT.format(
            ticker=state.company_ticker,
            doc_type=state.doc_type,
            sentiment=state.sentiment,
            sentiment_confidence=state.sentiment_confidence,
            metrics=metrics_str,
            claims=claims_str,
            entities=entities_str,
        )

    @staticmethod
    def _parse_llm_brief(raw: str, state: PipelineState) -> ResearchBrief:
        lines = raw.strip().split("\n")
        title = lines[0] if lines else "Research Brief"

        findings: list[str] = []
        risks: list[str] = []
        current_section = ""

        for line in lines[1:]:
            stripped = line.strip()
            lower = stripped.lower()

            if "key finding" in lower or "finding" in lower:
                current_section = "findings"
                continue
            elif "risk" in lower:
                current_section = "risks"
                continue
            elif "financial highlight" in lower or "highlight" in lower:
                current_section = ""
                continue

            if stripped.startswith(("-", "*", "•")):
                item = stripped.lstrip("-*• ").strip()
                if current_section == "findings":
                    findings.append(item)
                elif current_section == "risks":
                    risks.append(item)

        summary = " ".join(lines[1:4]) if len(lines) > 1 else ""

        ext = state.extraction
        return ResearchBrief(
            company_ticker=state.company_ticker,
            title=title.strip("# "),
            summary=summary[:500],
            key_findings=findings[:10],
            risk_factors=risks[:10],
            financial_highlights=ext.key_metrics if ext else {},
            citations=[
                {"claim": v.claim[:100], "verified": str(v.verified)}
                for v in state.verifications[:10]
            ],
            confidence_score=state.sentiment_confidence,
        )
