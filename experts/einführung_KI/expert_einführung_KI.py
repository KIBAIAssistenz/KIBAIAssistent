# experts/einführung_KI/expert_einführung_KI.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from services.llm_connector import llm
from experts.einführung_KI.prompt import PROMPT_EINFÜHRUNG_KI
from experts.einführung_KI.retriever_einführung_KI import make_einführung_ki_retriever
from safety.chain import build_safety_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate


def build_einführung_KI_expert(llm_generation=llm):
    """Erstellt den Experten für 'Einführung in die KI' inkl. Safety und Retriever."""
    retriever = make_einführung_ki_retriever()

    mq_prompt = ChatPromptTemplate.from_template(
        "Erzeuge mehrere Varianten der folgenden Frage in deutscher UND englischer Sprache.\n"
        "Gib nur die alternativen Fragen zurück, jeweils in einer neuen Zeile.\n\n"
        "Frage: {question}"
    )

    retriever_multi = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=mq_prompt,
    )

    chain = build_safety_chain(
        llm_generation=llm_generation,
        prompt=PROMPT_EINFÜHRUNG_KI,
        retriever=retriever_multi,
    )
    return {"chain": chain, "retriever": retriever, "module": "Einführung in die KI",}
