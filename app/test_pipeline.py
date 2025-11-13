from experts.einführung_KI.expert_einführung_KI import build_einführung_KI_expert
from experts.machine_learning.expert_ml import build_machine_learning_expert

ki = build_einführung_KI_expert()
print(ki["chain"].invoke("Hi Bimo, Was ist der Unterschied zwischen starker und schwacher KI?"))

# ml = build_machine_learning_expert()
# print(ml["chain"].invoke("Was bedeutet Klassifikation?."))


