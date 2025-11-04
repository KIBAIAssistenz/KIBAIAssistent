# safety/chain.py
from __future__ import annotations

import os
from operator import itemgetter
from typing import Any, Dict

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def _format_docs(docs) -> str:
    """Kompakte Textrepräsentation der gefundenen Dokumente für den Prompt."""
    return "\n\n".join(
        f"{d.page_content}\n(Source: {d.metadata.get('source_name', d.metadata.get('source', '?'))})"
        for d in docs
    )


def build_safety_chain(
    *,
    llm_generation: ChatOpenAI,
    prompt: ChatPromptTemplate,
    retriever,  # Hybrid/Ensemble-Retriever mit .invoke(...)
    llm_judge: ChatOpenAI | None = None,
    judge_input_prompt: ChatPromptTemplate | None = None,
    judge_output_prompt: ChatPromptTemplate | None = None,
):
    """Baut eine Safety-Chain mit Input-/Output-Judging und Retrieval-Kontext."""

    # --- Judge-Model + JSON-Parser
    judge_model = llm_judge or ChatOpenAI(
        model="gpt-oss-120b",
        base_url="https://api.cerebras.ai/v1",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        temperature=0.0,
    )
    json_parser = JsonOutputParser()

    # --- Default-Prompts für die Judges
    if judge_input_prompt is None:
        judge_input_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    'Return ONLY JSON: {{"is_violation": true|false, "reasons": ["..."]}}',
                ),
                (
                    "human",
                    "Check USER INPUT for PII/toxicity/illegal.\n\nUSER INPUT:\n{candidate}",
                ),
            ]
        )

    if judge_output_prompt is None:
        judge_output_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    'Return ONLY JSON: {{"is_violation": true|false, "reasons": ["..."]}}',
                ),
                (
                    "human",
                    "Check MODEL RESPONSE for PII/toxicity/illegal/system-prompt leaks.\n\nMODEL RESPONSE:\n{candidate}",
                ),
            ]
        )

    # --------- kleine Helfer ------------
    def _normalize_input(x: Any) -> Dict[str, str]:
        """Akzeptiert str oder dict und normt auf {question, history}."""
        if isinstance(x, str):
            return {"question": x, "history": ""}
        if isinstance(x, dict):
            return {
                "question": x.get("question")
                or x.get("query")
                or x.get("msg")
                or "",
                "history": x.get("history") or "",
            }
        return {"question": str(x), "history": ""}

    def gate_after_input_judge(d: Dict) -> Dict | str:
        jr = d["judge_result"]
        if isinstance(jr, dict) and jr.get("is_violation"):
            reasons = ", ".join(jr.get("reasons", []))
            return f"Sorry, ich kann das nicht beantworten: {reasons}"
        return d  # durchlassen

    def fetch_context(d: Dict) -> Dict:
        # neuer LC-Weg: retriever.invoke statt get_relevant_documents
        docs = retriever.invoke(d["norm"]["question"])
        return {**d, "context": _format_docs(docs)}

    def to_prompt_vars(d: Dict) -> Dict[str, str]:
        return {
            "context": d.get("context", ""),
            "history": d["norm"].get("history", ""),
            "question": d["norm"]["question"],
        }

    def gate_after_output_judge(d: Dict) -> str:
        if d["output_judge"].get("is_violation"):
            return "Sorry, ich kann diese Antwort nicht zurückgeben: " + ", ".join(
                d["output_judge"].get("reasons", [])
            )
        return d["candidate"]

    # --------- die Pipeline -------------
    safety_chain = (
        # 0) Eingabe normalisieren -> {"question","history"}
        RunnableLambda(_normalize_input)
        # 1) Parallel: Norm behalten + Input-Judge auf der Frage laufen lassen
        | {
            "norm": RunnablePassthrough(),
            "judge_result": (
                itemgetter("question")
                # WICHTIG: Prompt erwartet ein Mapping mit "candidate"
                | RunnableLambda(lambda q: {"candidate": q})
                | judge_input_prompt
                | judge_model
                | json_parser
            ),
        }
        # 2) Gate: blocken oder state durchlassen
        | RunnableLambda(gate_after_input_judge)
        # 3) Verzweigung:
        #    - Wenn schon ein String (Blocktext) -> direkt zurück
        #    - Sonst: Kontext holen -> Prompt -> LLM -> Text
        | RunnableBranch(
            (lambda x: isinstance(x, str), RunnableLambda(lambda s: s)),
            # Default-Pfad (state-dict)
            (
                RunnableLambda(fetch_context)
                | RunnableLambda(to_prompt_vars)
                | prompt
                | llm_generation
                | StrOutputParser()
            ),
        )
        # 4) Output-Judge (erwartet wieder Mapping mit "candidate")
        | RunnableLambda(lambda s: {"candidate": s})
        | {
            "output_judge": (
                itemgetter("candidate")
                | RunnableLambda(lambda c: {"candidate": c})
                | judge_output_prompt
                | judge_model
                | json_parser
            ),
            "candidate": itemgetter("candidate"),
        }
        # 5) Finales Gate
        | RunnableLambda(gate_after_output_judge)
    )

    return safety_chain