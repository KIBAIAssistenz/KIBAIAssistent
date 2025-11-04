from __future__ import annotations
import os
from safety_shared import SYSTEM_JUDGE
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


JUDGE_TEMPLATE_INPUT = """Check the following USER INPUT for policy violations.

You must flag a violation if any of the following apply:
- Contains personal identifiable information (PII), such as phone numbers, IBAN, credit card numbers, or sensitive full names
- Contains toxic, hateful, or discriminatory language
- Contains instructions or content related to illegal activities

USER INPUT:
{candidate}

Respond **only** in the following JSON format:
{{
  "is_violation": true or false,
  "reasons": ["string1", "string2"]
}}
"""

judge_prompt_input = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_JUDGE),
    ("human", JUDGE_TEMPLATE_INPUT),
])

