# experts/einführung_KI/expert_einführung_KI.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from services.llm_connector import llm
from experts.machine_learning.prompt_ml import PROMPT_MACHINE_LEARNING
from experts.machine_learning.retriever_ml import make_machine_learning_retriever
from safety.chain import build_safety_chain

def build_machine_learning_expert(llm_generation=llm):
    """Erstellt den Experten für 'Machine Learning' inkl. Safety und Retriever."""
    retriever = make_machine_learning_retriever()
    chain = build_safety_chain(
        llm_generation=llm_generation,
        prompt=PROMPT_MACHINE_LEARNING,
        retriever=retriever,
    )
    return {"chain": chain, "retriever": retriever,"module": "Machine Learning",}
