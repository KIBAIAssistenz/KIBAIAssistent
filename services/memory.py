# services/memory/memory.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  
 
from langchain.memory import ConversationEntityMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
 
 
#  PromptTemplate "memory" 
memory = PromptTemplate.from_template(
    """Antworte kurz.
Wenn der Nutzer nach seinem Namen oder persönlichen Infos fragt, nutze das gespeicherte Wissen und antworte ohne Erklärung, Beispiel, Typische Prüfungsfehler etc..
 
Gespräch:
{history}
Nutzer: {input}
Assistent:"""
)
 
# Tatsächliches Memory-Objekt (umbenannt) 
#entity_memory = ConversationEntityMemory(llm=llm)
 
# ConversationChain mit Template + Memory 
#conversation = ConversationChain(
    #llm=llm,
    #memory=entity_memory,
    #prompt=memory,       
    #verbose=True
#)
 
#def get_memory_chain():
    #"""Gibt eine ConversationChain mit Memory zurück."""
    #return conversation
 
 