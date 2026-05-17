import time

import structlog
from langgraph.graph import END, StateGraph

from src.agents.classifier import ClassifierAgent
from src.agents.extractor import ExtractorAgent
from src.agents.state import PipelineState
from src.agents.synthesizer import SynthesizerAgent
from src.agents.verifier import VerifierAgent
from src.models.pipeline import PipelineResult

log = structlog.get_logger()


def classify_node(state: dict) -> dict:
    ps = PipelineState(**state)
    ps = ClassifierAgent().run(ps)
    return ps.model_dump()


def extract_node(state: dict) -> dict:
    ps = PipelineState(**state)
    ps = ExtractorAgent().run(ps)
    return ps.model_dump()


def verify_node(state: dict) -> dict:
    ps = PipelineState(**state)
    ps = VerifierAgent().run(ps)
    return ps.model_dump()


def synthesize_node(state: dict) -> dict:
    ps = PipelineState(**state)
    ps = SynthesizerAgent().run(ps)
    return ps.model_dump()


def should_verify(state: dict) -> str:
    extraction = state.get("extraction")
    if extraction and extraction.get("claims"):
        return "verify"
    return "synthesize"


def build_pipeline_graph() -> StateGraph:
    graph = StateGraph(dict)

    graph.add_node("classify", classify_node)
    graph.add_node("extract", extract_node)
    graph.add_node("verify", verify_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "extract")
    graph.add_conditional_edges(
        "extract",
        should_verify,
        {"verify": "verify", "synthesize": "synthesize"},
    )
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)

    return graph


def run_pipeline(
    document_id: str,
    content: str,
    company_ticker: str = "",
    modality: str = "text",
) -> PipelineResult:
    start = time.time()

    initial_state = PipelineState(
        document_id=document_id,
        content=content,
        modality=modality,
        company_ticker=company_ticker,
    ).model_dump()

    graph = build_pipeline_graph()
    compiled = graph.compile()
    final_state = compiled.invoke(initial_state)

    elapsed = time.time() - start
    ps = PipelineState(**final_state)

    result = PipelineResult(
        document_id=document_id,
        extraction=ps.extraction,
        verifications=ps.verifications,
        brief=ps.brief,
        total_tokens_used=ps.tokens_used,
        total_cost_usd=ps.cost_usd,
        latency_seconds=round(elapsed, 2),
    )

    log.info("pipeline complete", document_id=document_id, latency=result.latency_seconds)
    return result
