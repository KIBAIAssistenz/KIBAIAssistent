from langchain.prompts import ChatPromptTemplate

INPUT_GUARD_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        'Du bist ein Sicherheits-Filter. Antworte IMMER nur mit gültigem JSON '
        'im Format {{ "is_violation": true|false, "reasons": ["..."] }}. '
        'Trage in "reasons" kurz die wichtigsten Gründe ein.'
    ),
    (
        "human",
        "Prüfe die folgende BENUTZER-EINGABE auf Verstösse gegen Sicherheitsregeln:\n"
        "- Gewalt, Extremismus oder Terrorismus\n"
        "- Hassrede oder Diskriminierung\n"
        "- sexuelle Inhalte mit Minderjährigen\n"
        "- Anleitungen zu Straftaten oder Selbstverletzung\n"
        "- Weitergabe sensibler persönlicher Daten (PII)\n\n"
        "BENUTZER-EINGABE:\n{candidate}"
    ),
])
