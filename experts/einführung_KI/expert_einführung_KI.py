# experts/einführung_KI/expert_einführung_KI.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from services.llm_connector import llm
from experts.einführung_KI.prompt import PROMPT_EINFÜHRUNG_KI
from experts.einführung_KI.retriever_einführung_KI import make_einführung_ki_retriever
from safety.chain import build_safety_chain

def build_einführung_KI_expert(llm_generation=llm):
    """Erstellt den Experten für 'Einführung in die KI' inkl. Safety und Retriever."""
    retriever = make_einführung_ki_retriever()
    chain = build_safety_chain(
        llm_generation=llm_generation,
        prompt=PROMPT_EINFÜHRUNG_KI,
        retriever=retriever,
        #return_source_documents=True,
    )
    return {"chain": chain, "retriever": retriever, "module": "Einführung in die KI",}
