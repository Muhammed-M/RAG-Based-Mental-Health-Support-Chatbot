# Mento: RAG-Based Mental Health Support Chatbot

Mento is a multilingual mental health support chatbot that combines local machine learning models, Groq-hosted LLM reasoning, and LangChain-powered retrieval augmented generation. The system is designed to understand user language, classify intent, detect emotional state, retrieve relevant counseling knowledge, and respond supportively in the user's verified language.

> Slogan: Your life matters.

## Important Safety Note

Mento is an educational NLP project and is not a replacement for a licensed mental health professional, medical care, or emergency support. If a user expresses immediate self-harm or suicide risk, the system returns a hardcoded crisis response and encourages contacting emergency or crisis services immediately.

## Key Features

- Multilingual language detection using a local TF-IDF based model.
- Crisis keyword detection with zero API cost.
- Single Groq intent-routing call for language verification, intent classification, translation, and English cleanup.
- Separate Groq keys for intent routing and RAG answer generation.
- Local emotion detection using the trained Module 2 model, with a memory-safe heuristic fallback.
- Conversational RAG with LangChain, Qdrant, and Hugging Face sentence embeddings.
- Persistent vector cache through Qdrant so the dataset is not embedded on every run.
- Response streaming through Flask Server-Sent Events.
- Conversation memory for follow-up mental health questions.
- LangSmith tracing support.
- Clean Flask front-end for local deployment.

## System Architecture

```text
User Query
    |
    v
Crisis Detection
    |-- crisis detected --> Hardcoded crisis response
    |
    v
Module 1: Language Detection
    |
    v
Intent Groq Call
    - verify or correct language
    - classify intent
    - translate non-English text to English
    - clean English typos and grammar
    |
    |-- greeting / goodbye / gratitude / system identity / out of scope
    |       --> polite direct response
    |
    v
Module 2: Emotion Detection
    |
    v
Module 4: Conversational RAG Retrieval
    |
    v
RAG Groq Synthesis
    - retrieved chunks
    - emotion
    - conversation memory
    - verified language
    |
    v
Post-Guardrails
    |
    v
Response to User
```

If Module 1 and the Groq intent router disagree about the language, Mento follows the Groq verified language.

## Repository Structure

```text
RAG-Based-Mental-Health-Support-Chatbot/
├── 1- Language Detection Model/
├── 2- Emotion Classifier Model/
├── 3- Intent Classifier Model/
├── Final System Files/
│   ├── Module 1/
│   ├── Module 2/
│   ├── Module 3/
│   └── Module 4/
│       ├── app.py
│       ├── mento_pipeline.py
│       ├── rag_service.py
│       ├── components.py
│       ├── prompts.py
│       ├── settings.py
│       ├── validate_module4.py
│       ├── requirements.txt
│       ├── Steps_to_use.txt
│       ├── templates/
│       └── static/
├── HF_Datasets_urls.md
├── NLP Final Project.pdf
└── README.md
```

## Modules

### Module 1: Language Detection

Uses a local TF-IDF based classifier to provide a fast language hint. This hint is passed to the Groq intent router, but it is not treated as final when the LLM verifies a different language.

### Module 2: Emotion Detection

Uses a local transformer-based emotion classifier trained for six labels:

- sadness
- joy
- love
- anger
- fear
- surprise

If Windows memory or pagefile availability is too low, Mento uses a lightweight heuristic fallback so the Flask app can continue running.

### Module 3: Intent Routing

The final system uses a LangChain + Groq implementation of the Module 3 routing logic:

- language verification
- intent classification
- translation to English
- typo and grammar cleanup

Supported high-level intents include:

- greeting
- goodbye
- gratitude
- system identity
- asking mental health question
- out of scope

### Module 4: RAG and Flask Application

Module 4 contains the final local application. It uses:

- Flask for the web interface.
- LangChain for the conversational RAG chain.
- Qdrant as the persistent vector database.
- Hugging Face dataset loading for the counseling conversation corpus.
- Sentence Transformers for embeddings.
- Groq for RAG response synthesis.
- LangSmith for optional tracing.

## Dataset

The RAG knowledge base uses:

```text
Amod/mental_health_counseling_conversations
https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
```

Embeddings are created with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The vector index is cached in Qdrant. If the configured Qdrant collection already contains vectors, Mento reuses it instead of downloading, chunking, and embedding the dataset again.

## Requirements

- Python 3.10 or newer recommended.
- Groq API key for intent routing.
- Groq API key for RAG response generation.
- Qdrant Cloud or local Qdrant instance.
- Optional LangSmith API key for tracing.
- Windows users should ensure sufficient pagefile/virtual memory for local model loading.

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

Recommended configuration:

```env
INTENT_GROQ_API_KEY=your_intent_groq_key
RAG_GROQ_API_KEY=your_rag_groq_key

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=Mental-Health

LANGSMITH_API_KEY=your_langsmith_key_optional
LANGSMITH_PROJECT=Mento-Module-4
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
```

Optional settings:

```env
GROQ_MODEL=llama-3.3-70b-versatile
INTENT_GROQ_MODEL=llama-3.3-70b-versatile
RAG_GROQ_MODEL=llama-3.3-70b-versatile

MENTO_DATASET_NAME=Amod/mental_health_counseling_conversations
MENTO_DATASET_SPLIT=train
MENTO_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MENTO_CHUNK_SIZE=900
MENTO_CHUNK_OVERLAP=120
MENTO_RETRIEVER_K=3

MENTO_FORCE_REBUILD_INDEX=false
MENTO_BUILD_INDEX_ON_STARTUP=false

MENTO_USE_LOCAL_EMOTION_MODEL=true
MENTO_MIN_AVAILABLE_PAGEFILE_MB=4096
MENTO_MIN_EMBEDDING_PAGEFILE_MB=1024

FLASK_RUN_HOST=127.0.0.1
FLASK_RUN_PORT=5000
FLASK_DEBUG=true
```

Do not commit real API keys to GitHub.

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

Run a local validation check:

```powershell
python validate_module4.py --router-debug
```

Run an optional end-to-end Groq + RAG test:

```powershell
python validate_module4.py --e2e
```

Expected behavior:

- Asking "What is your name?" returns that the system name is Mento.
- Crisis phrases return the hardcoded crisis response.
- Non-English mental health queries are translated to English for retrieval.
- The final response is written in the user's verified language.
- Short follow-ups such as "yes", "continue", or "نعم أريد" continue the previous mental health topic.
- The RAG chain retrieves Qdrant chunks and streams the generated answer.

## API Endpoints

### `GET /`

Renders the local Mento chat interface.

### `GET /api/health`

Returns system health, configured model names, Qdrant collection status, and local model settings.

### `POST /api/chat/stream`

Streams a chat response using Server-Sent Events.

Example payload:

```json
{
  "message": "I feel anxious and cannot sleep",
  "session_id": "demo-session"
}
```

### `POST /api/chat/clear`

Clears server-side chat memory for a session.

### `POST /api/index/rebuild`

Rebuilds the Qdrant vector index from the Hugging Face dataset.

## Troubleshooting

### The first RAG query is slow

The first run may load the embedding model, connect to Qdrant, or build the vector index. Later runs reuse the cached Qdrant collection.

### Windows says the paging file is too small

The local emotion model is large. Close memory-heavy applications or increase Windows virtual memory. Mento can also use a lighter fallback:

```env
MENTO_USE_LOCAL_EMOTION_MODEL=false
```

### The browser shows a network error

Check that Flask is running and that Groq and Qdrant are reachable. Restart the app if the local process stopped.

### The response is not in the expected language

Mento uses the Groq verified language from the intent-routing call, not only the Module 1 hint. Check the routing output with:

```powershell
python validate_module4.py --router-debug
```

## Technologies Used

- Python
- Flask
- LangChain
- LangChain Groq
- LangChain Qdrant
- Qdrant
- Hugging Face Datasets
- Sentence Transformers
- Transformers
- PyTorch
- scikit-learn
- LangSmith

## Contributors

This project was developed as a modular NLP final project with separate workstreams for language detection, emotion classification, intent routing, and RAG-based support generation.
