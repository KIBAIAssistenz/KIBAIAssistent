# import os
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# # --- Initialisiere LLM explizit für CEREBRAS ---
# assert "CEREBRAS_API_KEY" in os.environ, "CEREBRAS_API_KEY fehlt in den Env Vars!"

# llm = ChatOpenAI( #Cerebras Verbindung
#     model="gpt-oss-120b",   
#     api_key=os.environ["CEREBRAS_API_KEY"],
#     base_url="https://api.cerebras.ai/v1",
#     temperature=0.3,
# )

# print("Sende Test-Ping...")
# try:
#     msg = llm.invoke("Sag exakt: pong")
#     print("Antworttyp:", type(msg))
#     # msg ist i.d.R. ein AIMessage – gib Inhalt sicher aus:
#     print("Inhalt:", getattr(msg, "content", msg))
# except Exception as e:
#     print("FEHLER beim LLM-Aufruf:", repr(e))


# import os
# import langchain
# from dotenv import load_dotenv
# load_dotenv()

# from langchain_openai import ChatOpenAI


# #LLM_MODEL = "openai/gpt-oss-20b:free"
# LLM_MODEL = "gpt-oss-120b"
# LLM_TEMPERATURE = 0.3
# BASE_URL = "https://api.cerebras.ai/v1"
# OPENROUTER_API_KEY = os.getenv("CEREBRAS_API_KEY")
# USER_PROMPT="Ich verstehe GenAI nicht, kannst du das mir einfach erklären?"


# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate


# llm = ChatOpenAI(
#     model=LLM_MODEL,
#     temperature=LLM_TEMPERATURE,
#     base_url=BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# print(type(llm))
