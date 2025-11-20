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

from .judge_input import INPUT_GUARD_PROMPT
from .judge_output import OUTPUT_GUARD_PROMPT

def _format_history(history) -> str:
    """Macht aus dem Chat-Verlauf einen kompakten Text für den Prompt."""
    if not history:
        return ""

    # Verlauf als Liste von (user_msg, assistant_msg)
    if isinstance(history, list) and history and isinstance(history[0], (list, tuple)):
        turns = []
        for user_msg, assistant_msg in history[-3:]:
            if user_msg:
                turns.append(f"User: {user_msg}")
            if assistant_msg:
                turns.append(f"Assistent: {assistant_msg}")
        return "\n".join(turns)

    # Verlauf als Liste von Dicts mit 'role' und 'content'
    if isinstance(history, list) and history and isinstance(history[0], dict) and "role" in history[0]:
        turns = []
        for msg in history[-6:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                turns.append(f"User: {content}")
            else:
                turns.append(f"Assistent: {content}")
        return "\n".join(turns)

    # Fallback
    return str(history)

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

    judge_input_prompt = judge_input_prompt or INPUT_GUARD_PROMPT
    judge_output_prompt = judge_output_prompt or OUTPUT_GUARD_PROMPT

    if judge_input_prompt is None:
        raise ValueError("judge_input_prompt must be provided")
    if judge_output_prompt is None:
        raise ValueError("judge_output_prompt must be provided")

    # --- Prompt für HUMORVOLLE Absage, wenn Input-Guard triggert
    refusal_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Du bist ein freundlicher, humorvoller KI-Assistent. "
                    "Die ursprüngliche Nutzeranfrage war aus Sicherheitsgründen problematisch. "
                    "Du DARFST keine Details zur Anfrage wiederholen und keine Anleitungen "
                    "zu gefährlichen oder verbotenen Dingen geben.\n\n"
                    "Schreibe eine kurze, lockere, humorvolle, aber klare Absage auf Deutsch. "
                    "Maximal 3 Sätze. Kein Fachchinesisch."
                ),
            ),
            (
                "human",
                (
                    "Die Anfrage wurde blockiert aus folgenden Gründen (nur für dich zur Einordnung):\n"
                    "{reasons}\n\n"
                    "Formuliere bitte eine passende humorvolle Absage."
                ),
            ),
        ]
    )

    # --------- kleine Helfer ------------
    def _normalize_input(x: Any) -> Dict[str, str]:
        """Akzeptiert str oder dict und normt auf {question, history}."""
        if isinstance(x, str):
            return {"question": x, "history": ""}

        if isinstance(x, dict):
            raw_history = x.get("history") or ""
            return {
                "question": x.get("question") or x.get("query") or x.get("msg") or "",
                "history": _format_history(raw_history),
            }

        return {"question": str(x), "history": ""}

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

    def gate_after_input_judge(d: Dict) -> Dict:
        """Markiert nur, ob geblockt werden soll – Antwort kommt später vom LLM."""
        jr = d["judge_result"]
        is_blocked = bool(isinstance(jr, dict) and jr.get("is_violation"))
        return {**d, "blocked": is_blocked}

    def gate_after_output_judge(d: Dict) -> str:
        """Falls Output-Guard triggert, kurze Standard-Absage."""
        if d["output_judge"].get("is_violation"):
            return (
                "Ups, diese Antwort kann ich dir so nicht geben – "
                "meine Sicherheitsregeln funken dazwischen. 🙈"
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
            # 3a) FALL A: Input wurde blockiert -> humorvolle Absage
            (
                lambda d: isinstance(d, dict) and d.get("blocked", False),
                (
                    RunnableLambda(
                        lambda d: {
                            "reasons": ", ".join(d["judge_result"].get("reasons", []))
                            or "Verstoss gegen Sicherheitsregeln"
                        }
                    )
                    | refusal_prompt
                    | llm_generation
                    | StrOutputParser()
                ),
            ),
            # 3b) FALL B: alles ok -> normaler RAG-Flow
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