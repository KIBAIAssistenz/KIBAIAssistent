# experts/bis/expert_bis.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from services.llm_connector import llm
from experts.bis.prompt_bis import PROMPT_BIS
from experts.bis.retriever_bis import make_bis_retriever
from safety.chain import build_safety_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate


def build_bis_expert(llm_generation=llm):
    """Erstellt den Experten für 'Betriebliche Informationssysteme (BIS)' inkl. Safety und Retriever."""
    
    # 1. Hybrid-Retriever (BM25 + FAISS)
    retriever = make_bis_retriever()

    # 2. Multi-Query Retriever Prompt
    mq_prompt = ChatPromptTemplate.from_template(
        "Erzeuge mehrere Varianten der folgenden Frage in deutscher UND englischer Sprache.\n"
        "Gib nur die alternativen Fragen zurück, jeweils in einer neuen Zeile.\n\n"
        "Frage: {question}"
    )

    # 3. Multi-Query Retriever
    retriever_multi = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=mq_prompt,
    )

    # 4. Safety Chain + Prompt + Retriever
    chain = build_safety_chain(
        llm_generation=llm_generation,
        prompt=PROMPT_BIS,
        retriever=retriever_multi,
    )

    # Rückgabe im selben Format wie ML
    return {
        "chain": chain,
        "retriever": retriever,
        "module": "Betriebliche Informationssysteme (BIS)",
    }
