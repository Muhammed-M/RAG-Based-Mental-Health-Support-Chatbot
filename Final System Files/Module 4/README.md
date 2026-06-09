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

> Your Life Matters

> **Safety note:** Mento is an educational NLP project. It is **not** a replacement
> for a licensed mental health professional or emergency support. If a user expresses
> immediate self-harm or suicide risk, the system returns a hardcoded crisis response
> in English, Arabic, Spanish, French, Chinese, or Hindi, encourages emergency or
> crisis support immediately, then continues with a brief safety-aware RAG follow-up.

## Required Space Secrets

Configure these under **Settings → Secrets** in your Hugging Face Space:

| Secret | Required | Notes |
| --- | --- | --- |
| `INTENT_GROQ_API_KEY` | yes | Used for language verification, intent classification, translation, typo cleanup |
| `RAG_GROQ_API_KEY` | yes | Used for RAG answer generation and post-guardrail LLM check |
| `QDRANT_URL` | yes | Persistent vector DB endpoint |
| `QDRANT_API_KEY` | yes | Qdrant API key |
| `QDRANT_COLLECTION` | no | Defaults to `Mental-Health-Counseling-Embeddings` |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | for feedback logging | Service account JSON used to append thumbs feedback rows to Google Sheets |
| `FEEDBACK_GOOGLE_SHEET_ID` | no | Defaults to `1Ja4NI2w9YbG6rw623bRDoamtm8YrSmeaChJ44xeuiyY` |
| `FEEDBACK_GOOGLE_WORKSHEET_GID` | no | Defaults to `0`, the worksheet id in the provided link |
| `FEEDBACK_APPS_SCRIPT_URL` | recommended for feedback logging | Google Apps Script web app `/exec` URL. When set, Mento uses this instead of service account access |
| `FEEDBACK_APPS_SCRIPT_SECRET` | no | Optional shared secret sent to the Apps Script webhook |
| `MENTO_EMBEDDING_MODEL` | no | Defaults to `sentence-transformers/all-MiniLM-L6-v2` |
| `MENTO_USE_LOCAL_EMOTION_MODEL` | no | Defaults to `false` on the Space (set `true` to enable the local PyTorch model) |
| `LANGSMITH_API_KEY` | no | Enables LangSmith tracing |
| `LANGSMITH_PROJECT` | no | Defaults to `Mental Health Rag ChatBot` |

Recommended feedback logging: paste `apps_script_feedback_webapp.gs` into a Google
Apps Script project bound to the feedback Sheet, deploy it as a web app, and put its
`/exec` URL in `FEEDBACK_APPS_SCRIPT_URL`. Mento only writes a row with `Query`,
`Response`, and `Like/Dislike` after a user clicks a thumbs up/down button.

Fallback feedback logging: if `FEEDBACK_APPS_SCRIPT_URL` is empty, Mento tries the
older Google service-account flow. In that mode, share the Google Sheet with the
service account `client_email` as an editor.

Retrieved chunks are visible in LangSmith on `Mento.RAG.answer_from_retrieved_chunks`
and `Mento.guardrail.check_response_with_chunks` as `retrieved_chunks` input/metadata.

The first time the Space boots, Mento will build the Qdrant vector index from
`Amod/mental_health_counseling_conversations`. After that the cached Qdrant collection
is reused.

## Local development

See the parent repository's `README.md` and `Final System Files/Module 4/Steps_to_use.txt`.
