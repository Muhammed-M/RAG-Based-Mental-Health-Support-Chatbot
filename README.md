# Mento: RAG-Based Mental Health Support Chatbot

Mento is a multilingual mental health support chatbot that combines local machine
learning models, Groq-hosted LLM reasoning, LangChain retrieval augmented generation,
Qdrant vector search, and a Flask chat interface.

The system detects the user's language, verifies intent, translates or cleans the
message for retrieval, detects emotional state, retrieves relevant counseling
knowledge, applies response guardrails, and answers in the verified user language.

> Slogan: Your Life Matters

## Important Safety Note

Mento is an educational NLP project. It is not a replacement for licensed mental
health care, medical advice, emergency support, or a crisis line.

If a user expresses direct or immediate self-harm or suicide intent, Mento returns a
hardcoded crisis safety response before any generated support. Crisis responses are
localized for English, Arabic, Spanish, French, Hindi, and Chinese. Other detected
languages fall back to English for the crisis safety message.

## Current Features

- Local Module 1 language detection using TF-IDF model artifacts.
- Groq-based language verification that can correct the local language hint.
- Support for 20 language codes in routing: `ar`, `bg`, `de`, `el`, `en`, `es`,
  `fr`, `hi`, `it`, `ja`, `nl`, `pl`, `pt`, `ru`, `sw`, `th`, `tr`, `ur`, `vi`,
  and `zh`.
- Direct crisis detection before and after LLM routing, with negated and historical
  suicide references handled separately from immediate crisis intent.
- One Groq intent-routing call for intent classification, language verification,
  translation to English, and English typo/grammar cleanup.
- Separate Groq keys for intent routing and RAG generation, with `GROQ_API_KEY`
  retained as a backward-compatible fallback.
- Direct handling for greetings, goodbye, gratitude, system identity, and
  out-of-scope prompts.
- Local Module 2 transformer emotion detection for sadness, joy, love, anger, fear,
  and surprise.
- Memory-safe heuristic emotion fallback when the local model is disabled or Windows
  pagefile availability is too low.
- High-distress flagging for sadness, anger, and fear when confidence is high.
- Conversational RAG with LangChain, a history-aware retriever, Qdrant, and Hugging
  Face sentence embeddings.
- Persistent Qdrant vector cache so the counseling dataset is not embedded on every
  startup.
- Optional index rebuild on startup, manual rebuild endpoint, named Qdrant vectors,
  and dataset row limiting for faster tests.
- Streaming chat responses through Flask Server-Sent Events.
- Server-side conversation memory and browser-side persistent session IDs.
- Contextual follow-up handling for short replies such as "yes", "continue", and
  Arabic affirmative replies.
- Localized suggested follow-up question chips in the web UI.
- Clear-chat workflow that resets server memory and browser session state.
- Post-response guardrails using both heuristics and a Groq LLM check.
- Safe fallback responses when emotion detection, retrieval, embeddings, Groq, or
  Qdrant resources are temporarily unavailable.
- LangSmith tracing support for retrieval, RAG answer generation, guardrail checks,
  retrieved chunk metadata, source rows, scores, and content previews.
- Thumbs up/down feedback controls in the UI.
- Feedback logging to Google Sheets through a preferred Google Apps Script webhook
  or a fallback Google service-account flow.
- Dockerfile for Hugging Face Spaces or container deployment on port `7860`.
- Health endpoint exposing model, Qdrant, LangSmith, and feedback configuration
  status.
- Validation script for local wiring checks and optional end-to-end Groq + RAG tests.

## System Architecture

```text
User message
  |
  v
Direct crisis guard
  |-- direct/immediate crisis -> hardcoded safety response
  |                            -> crisis-aware support follow-up
  |
  v
Module 1 local language hint
  |
  v
Groq intent router
  - verify/correct language
  - classify intent
  - translate non-English text to English
  - clean English typos and grammar
  |
  |-- greeting/goodbye/gratitude/system identity/out of scope
  |      -> direct generated response
  |
  v
Module 2 emotion detection
  |
  v
Qdrant retrieval with LangChain
  |
  v
Groq RAG synthesis
  - retrieved chunks
  - emotion and distress flag
  - conversation memory
  - verified language
  |
  v
Heuristic + LLM guardrails
  |
  v
SSE streamed response to the web UI
```

If Module 1 and the Groq intent router disagree about the language, Mento follows
the Groq verified language.

## Repository Structure

```text
RAG-Based-Mental-Health-Support-Chatbot/
|-- 2- Emotion Classifier Model/
|-- 3- Intent Classifier Model/
|-- Final System Files/
|   |-- Module 1/
|   |-- Module 2/
|   |-- Module 3/
|   `-- Module 4/
|       |-- app.py
|       |-- mento_pipeline.py
|       |-- rag_service.py
|       |-- components.py
|       |-- prompts.py
|       |-- feedback_service.py
|       |-- settings.py
|       |-- validate_module4.py
|       |-- apps_script_feedback_webapp.gs
|       |-- Dockerfile
|       |-- requirements.txt
|       |-- pyproject.toml
|       |-- Steps_to_use.txt
|       |-- templates/
|       `-- static/
|-- HF_Datasets_urls.md
|-- NLP Final Project.pdf
`-- README.md
```

## Modules

### Module 1: Language Detection

Module 1 provides a fast local language hint using saved TF-IDF/vectorizer and
classifier artifacts. Module 4 can load either the notebook-era filenames or the
current final-system filenames.

The language hint is passed to the Groq router, but the LLM verification step is
treated as authoritative when it disagrees with the local model.

### Module 2: Emotion Detection

Module 2 detects one of six emotions:

- sadness
- joy
- love
- anger
- fear
- surprise

When `USE_LOCAL_EMOTION_MODEL=false`, or when Windows pagefile availability is below
`MIN_AVAILABLE_PAGEFILE_MB`, Mento uses a lightweight heuristic fallback instead of
loading the larger local transformer model.

### Module 3: Intent Routing

The final system implements Module 3 routing with LangChain and Groq. The router:

- verifies the true language code,
- classifies the user intent,
- translates non-English messages to clean English for retrieval,
- cleans English typos and grammar while preserving emotional urgency.

Supported high-level intents are:

- `greeting`
- `goodbye`
- `gratitude`
- `system_identity`
- `asking_mental_health_question`
- `out_of_scope`

### Module 4: RAG, Safety, Feedback, and Flask UI

Module 4 is the final runnable application. It includes:

- Flask web routes and Server-Sent Events streaming.
- LangChain conversational RAG.
- Qdrant vector storage and retrieval.
- Hugging Face dataset loading and chunking.
- Sentence Transformers embeddings.
- Groq answer generation.
- Heuristic and LLM response guardrails.
- LangSmith observability.
- Google Sheets feedback logging.
- A responsive local chat UI with suggestions, feedback buttons, status updates,
  persistent sessions, and clear-chat support.

## Dataset and Embeddings

The default RAG knowledge base is:

```text
Amod/mental_health_counseling_conversations
https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
```

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Mento chunks the counseling dataset into Qdrant documents. If the configured Qdrant
collection already exists and contains vectors, Mento reuses it instead of rebuilding
the index.

## Requirements

- Python 3.12 is the current Module 4 target.
- Groq API access for routing and answer generation.
- Qdrant Cloud or a local Qdrant instance.
- Optional LangSmith API key for tracing.
- Optional Google Sheets or Apps Script setup for feedback logging.
- Windows users should keep enough pagefile/virtual memory available for local
  transformer and embedding model loading.

## Installation

Open a terminal in the final Module 4 directory:

```powershell
cd "D:\Python\RAG-Based-Mental-Health-Support-Chatbot\Final System Files\Module 4"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in:

```text
Final System Files/Module 4/.env
```

Do not commit real API keys, Qdrant credentials, LangSmith credentials, Google
service-account JSON, or Apps Script secrets.

### Required Core Settings

```env
INTENT_GROQ_API_KEY=your_intent_groq_key
RAG_GROQ_API_KEY=your_rag_groq_key

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=Mental-Health
```

Backward-compatible Groq fallback:

```env
GROQ_API_KEY=your_single_groq_key
```

If `INTENT_GROQ_API_KEY` or `RAG_GROQ_API_KEY` is missing, Mento falls back to
`GROQ_API_KEY`.

### Optional Model and RAG Settings

```env
GROQ_MODEL=llama-3.3-70b-versatile
INTENT_GROQ_MODEL=llama-3.3-70b-versatile
RAG_GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.3
GROQ_MAX_TOKENS=700

DATASET_NAME=Amod/mental_health_counseling_conversations
DATASET_SPLIT=train
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=900
CHUNK_OVERLAP=120
RETRIEVER_K=4
MAX_DATASET_ROWS=0

QDRANT_VECTOR_NAME=
FORCE_REBUILD_INDEX=false
BUILD_INDEX_ON_STARTUP=false
```

### Optional Memory Settings

```env
USE_LOCAL_EMOTION_MODEL=true
MIN_AVAILABLE_PAGEFILE_MB=4096
MIN_EMBEDDING_PAGEFILE_MB=0
```

Set `USE_LOCAL_EMOTION_MODEL=false` to force the heuristic emotion fallback.

### Optional LangSmith Settings

```env
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=Mento-Module-4
LANGSMITH_TRACING=true
LANGSMITH_TRACING_V2=true
LANGCHAIN_TRACING_V2=true
```

When `LANGSMITH_API_KEY` is present, `settings.py` also enables the tracing
environment flags automatically.

### Optional Feedback Logging Settings

Preferred Google Apps Script webhook mode:

```env
FEEDBACK_APPS_SCRIPT_URL=https://script.google.com/macros/s/.../exec
FEEDBACK_APPS_SCRIPT_SECRET=optional_shared_secret
FEEDBACK_APPS_SCRIPT_TIMEOUT=10
```

Fallback Google service-account mode:

```env
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account", "...":"..."}
GOOGLE_SERVICE_ACCOUNT_JSON_B64=
GOOGLE_SERVICE_ACCOUNT_FILE=
GOOGLE_APPLICATION_CREDENTIALS=

FEEDBACK_GOOGLE_SHEET_ID=your_sheet_id
FEEDBACK_GOOGLE_WORKSHEET_GID=0
FEEDBACK_GOOGLE_WORKSHEET_NAME=
```

The checked-in helper script is:

```text
Final System Files/Module 4/apps_script_feedback_webapp.gs
```

Mento appends only three feedback columns after a user clicks thumbs up or thumbs
down:

```text
Query | Response | Like/Dislike
```

### Optional Flask Settings

```env
FLASK_RUN_HOST=127.0.0.1
FLASK_RUN_PORT=5000
FLASK_DEBUG=true
```

## Running the Application

From `Final System Files/Module 4`:

```powershell
python app.py
```

Open the local app:

```text
http://127.0.0.1:5000
```

Health check:

```text
GET http://127.0.0.1:5000/api/health
```

## Validation

Run a local wiring check:

```powershell
python validate_module4.py --router-debug
```

Run an optional end-to-end Groq + RAG test:

```powershell
python validate_module4.py --e2e
```

Expected behavior:

- Asking "What is your name?" returns that the system name is Mento.
- Direct/immediate crisis phrases return the hardcoded safety response before any
  generated support.
- Historical or negated suicide mentions continue through normal mental-health RAG
  handling instead of triggering the direct crisis route.
- Non-English mental health queries are translated to English for retrieval.
- Final responses are written in the verified user language.
- Short follow-ups such as "yes", "continue", or Arabic affirmative replies continue
  the previous mental health topic.
- The RAG chain retrieves Qdrant chunks and streams generated tokens.
- Guardrails can revise unsafe or hallucinated responses.

## API Endpoints

### `GET /`

Renders the local Mento chat interface.

### `GET /favicon.ico`

Returns an empty `204` response.

### `GET /api/health`

Returns system health, configured model names, Qdrant collection status, local model
settings, LangSmith project name, and feedback logging configuration status.

### `POST /api/chat/stream`

Streams a chat response using Server-Sent Events.

Example payload:

```json
{
  "message": "I feel anxious and cannot sleep",
  "session_id": "demo-session",
  "last_mental_health_topic": ""
}
```

Common event types include:

- `session`
- `metadata`
- `token`
- `new_assistant_message`
- `notice`
- `error`
- `done`

### `POST /api/chat/clear`

Clears server-side chat memory and the last tracked mental health topic for a session.

Example payload:

```json
{
  "session_id": "demo-session"
}
```

### `POST /api/feedback`

Logs thumbs feedback for an answer.

Example payload:

```json
{
  "query": "I feel anxious and cannot sleep",
  "response": "I hear you...",
  "feedback": "like"
}
```

Accepted feedback values include `like`, `thumbs_up`, `up`, `dislike`,
`thumbs_down`, and `down`.

### `POST /api/index/rebuild`

Rebuilds the Qdrant vector index from the configured Hugging Face dataset.

For public deployments, protect this endpoint behind trusted access controls.

## Hugging Face Spaces and Docker

Module 4 includes a Dockerfile for container deployment. The container:

- uses Python 3.12 slim,
- installs Module 4 requirements and `gunicorn`,
- listens on `0.0.0.0:7860`,
- runs one Gunicorn worker with multiple threads for SSE streaming,
- disables the local emotion model by default with `USE_LOCAL_EMOTION_MODEL=false`.

To enable the local emotion model in a Space, set `USE_LOCAL_EMOTION_MODEL=true` as a
Space secret or environment variable.

Configure required secrets in the Space settings:

- `INTENT_GROQ_API_KEY`
- `RAG_GROQ_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- optional LangSmith settings
- optional feedback logging settings

The first Space boot may build the Qdrant vector index. Later boots reuse the cached
Qdrant collection.

## LangSmith Observability

When LangSmith tracing is enabled, retrieved chunk details are attached to relevant
runs, including source, score, metadata, content preview, and content length.

Useful run names:

- `Mento.retrieve_chunks`
- `Mento.RAG.answer_from_retrieved_chunks`
- `Mento.guardrail.check_response_with_chunks`

## Troubleshooting

### The first RAG query is slow

The first run may load the embedding model, connect to Qdrant, download the dataset,
or build the vector index. Later runs reuse the cached Qdrant collection.

### Windows says the paging file is too small

The local transformer models can need substantial virtual memory. Close memory-heavy
applications or increase Windows virtual memory. You can also force the lighter
emotion fallback:

```env
USE_LOCAL_EMOTION_MODEL=false
```

### The browser shows a network error

Check that Flask is running and that Groq and Qdrant are reachable. Restart the app
if the local process stopped.

### Feedback is not saved

Check `/api/health` to confirm whether Apps Script, Google Sheet, and credential
settings are configured. If using Apps Script, confirm that the deployed `/exec` URL
is set in `FEEDBACK_APPS_SCRIPT_URL`.

### The response is not in the expected language

Mento uses the Groq verified language from the routing call, not only the Module 1
hint. Check router behavior with:

```powershell
python validate_module4.py --router-debug
```

### The index rebuild keeps running

Confirm that `FORCE_REBUILD_INDEX=false` after the rebuild is complete. Use
`BUILD_INDEX_ON_STARTUP=true` only when you intentionally want startup-time index
validation or rebuilding.

## Technologies Used

- Python
- Flask
- Gunicorn
- LangChain
- LangChain Groq
- LangChain Qdrant
- Qdrant
- Hugging Face Datasets
- Sentence Transformers
- Transformers
- PyTorch
- scikit-learn
- Google Apps Script
- gspread
- LangSmith

## Contributors

This project was developed as a modular NLP final project with separate workstreams
for language detection, emotion classification, intent routing, and RAG-based support
generation.
