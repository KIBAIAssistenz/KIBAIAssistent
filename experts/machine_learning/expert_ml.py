# experts/einführung_KI/expert_einführung_KI.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from services.llm_connector import llm
from experts.machine_learning.prompt_ml import PROMPT_MACHINE_LEARNING
from experts.machine_learning.retriever_ml import make_machine_learning_retriever
from safety.chain import build_safety_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate

def build_machine_learning_expert(llm_generation=llm):
    """Erstellt den Experten für 'Machine Learning' inkl. Safety und Retriever."""
    retriever = make_machine_learning_retriever()

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
        prompt=PROMPT_MACHINE_LEARNING,
        retriever=retriever_multi,
    )
    return {"chain": chain, "retriever": retriever,"module": "Machine Learning",}
