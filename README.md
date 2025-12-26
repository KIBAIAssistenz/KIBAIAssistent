
---
# 📘 KI-gestützter Lernassistent für den Studiengang *Business AI*

Ein interaktiver KI-Tutor, optimiert für den Studiengang **Business AI**, der Studierende bei den Vorlesungen **Künstliche Intelligenz** und **Machine Learning** unterstützt.
Der Assistent nutzt ein **Expertenagenten-System**, **RAG**, **Safeguards**, **Memory** und **RLHF (Star Ratings)**, um präzise, sichere und personalisierte Lernunterstützung zu bieten.

---

## 🚀 Features

### 🤖 Mehrere Experten-Agenten

Spezialisierte Agenten für unterschiedliche Module aus dem Studiengang:

* **Einführung in die KI**
* **Maschinelles Lernen**
* **Betriebliche Informationssysteme**

---

### 📚 RAG – Retrieval Augmented Generation

* Nutzung von **FAISS** & **BM25** für hybride Suche
* PDF-Verarbeitung mit **PyPDFLoader**
* Generierung von kontextbezogenen Antworten basierend auf Vorlesungsfolien, Zusammenfassungen & Skripten
* Unterstützung für grosse Dokumente, Kapitelweise Chunks, Scoring & Re-Ranking

---

### 🧠 Memory pro Agent

* Jeder Chatverlauf besitzt seinen eigenen Memory-Kontext
* Nachverfolgung Fragen innerhalb des Chats
* Verbesserung der Antwortqualität über Zeit

---

### 🛡 Input/Output Safeguards

* Validierung der Nutzereingaben (Input Guard)
* Sicherheitsschicht für generierte Antworten (Output Guard)
* Anpassbarer Ton (formell, locker, humorvoll)
* Schützt vor Halluzinationen & Fehlverhalten

---

### 🔗 LangChain Experten-Chains

* Modularer Aufbau
* Jede Expertenkette besteht aus:

  * **Guard → Retrieval → LLM → RLHF-Rating**
* Leicht erweiterbar für neue Kurse oder Module

---

### 🌐 Web UI (Flask)

* FHNW-orientierte Weboberfläche
* Auswahl des passenden Experten über DropDown Menü
* Chat-Historie pro Session
* Visuelle Ausgabe im Browser
* Leicht integrierbar in Hochschul-Tools

---

### 🛠 Modelle

Unterstützung für verschiedene LLM-Anbieter:

* **Cerebras (kostenlose LLMs über Einbindung der API)**

---

## 📦 Installation & Setup

### Voraussetzungen

* **Python 3.13.2**
* **VS Code** oder **PyCharm**
* Optional: API Keys für Cerebras oder Hugging Face
* Zugriff auf die PDF-Daten und RAG-Ressourcen

---

## 🔧 How to use

### 1️⃣ Repository klonen

```bash
git clone https://github.com/KIBAIAssistenz/KIBAIAssistent
cd KIBAIAssistent
```

---

### 2️⃣ Virtuelle Umgebung erstellen

```bash
python -m venv ./.venv
```

---

### 3️⃣ Virtuelle Umgebung aktivieren

#### Windows:

```bash
.\.venv\Scripts\Activate
```

#### macOS:

```bash
source .venv/bin/Activate
```

---

### 4️⃣ Dependencies installieren

#### Windows:

```bash
pip install -r .\requirements.txt
```

#### macOS:

```bash
pip install -r ./requirements.txt
```

---

### 5️⃣ Anwendung starten

```bash
python app/UI_kerstin.py
```

Danach läuft der KI-Assistent unter:

```
http://localhost:5000
```

---

## 🧱 Projektstruktur (Kurzüberblick)

```
KIBAIAssistent/
│
├── app/UI_kerstin.py        # Flask Web UI
├── experts/                 # Alle Experten-Module
│   ├── einführung_KI/
│   ├── machine_learning/
│   └── ...
│
├── rag/                     # Retrieval Pipeline (FAISS, BM25)
│
├── safeguards/              # Input/Output Guards
│
├── memory/                  # Memory pro Agent
│
├── services/                # Schnittstellen zu LLMs & Tools
│
├── data/                    # PDFs, Chunks, Vektordatenbanken
│
├── config.py                # API Keys & Einstellungen
└── requirements.txt
```

---

## 🧠 Technologien

* **Python 3.13.2**
* **LangChain**
* **FAISS**
* **BM25**
* **RAG Pipeline**
* **dotenv**
* **PyPDFLoader**
* **Textstat**
* **Flask**
* **OpenAI / Cerebras / HuggingFace APIs**
