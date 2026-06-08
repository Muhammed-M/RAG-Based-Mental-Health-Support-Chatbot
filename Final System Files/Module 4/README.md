---
title: Mento
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RAG-based multilingual mental health support chatbot
---

# Mento — RAG-Based Mental Health Support Chatbot

A multilingual mental health support chatbot combining local ML models, Groq-hosted
LLM reasoning, and LangChain-powered retrieval augmented generation. Built as a final
project for the ITI NLP & LLM track.

> Your life matters.

> **Safety note:** Mento is an educational NLP project. It is **not** a replacement
> for a licensed mental health professional or emergency support. If a user expresses
> immediate self-harm or suicide risk, the system returns a hardcoded crisis response
> and encourages contacting emergency or crisis services immediately.

## Required Space Secrets

Configure these under **Settings → Secrets** in your Hugging Face Space:

| Secret | Required | Notes |
| --- | --- | --- |
| `INTENT_GROQ_API_KEY` | yes | Used for language verification, intent classification, translation, typo cleanup |
| `RAG_GROQ_API_KEY` | yes | Used for RAG answer generation and post-guardrail LLM check |
| `QDRANT_URL` | yes | Persistent vector DB endpoint |
| `QDRANT_API_KEY` | yes | Qdrant API key |
| `QDRANT_COLLECTION` | no | Defaults to `Mental-Health-Counseling-Embeddings` |
| `MENTO_EMBEDDING_MODEL` | no | Defaults to `sentence-transformers/all-MiniLM-L6-v2` |
| `MENTO_USE_LOCAL_EMOTION_MODEL` | no | Defaults to `false` on the Space (set `true` to enable the local PyTorch model) |
| `LANGSMITH_API_KEY` | no | Enables LangSmith tracing |
| `LANGSMITH_PROJECT` | no | Defaults to `Mental Health Rag ChatBot` |

The first time the Space boots, Mento will build the Qdrant vector index from
`Amod/mental_health_counseling_conversations`. After that the cached Qdrant collection
is reused.

## Local development

See the parent repository's `README.md` and `Final System Files/Module 4/Steps_to_use.txt`.
