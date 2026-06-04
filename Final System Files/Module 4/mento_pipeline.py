from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from components import (
    CRISIS_RESPONSE,
    CrisisDetector,
    EmotionDetector,
    LanguageDetector,
    heuristic_guardrail,
    is_followup_affirmation,
    is_system_identity_question,
)
from prompts import DIRECT_RESPONSE_SYSTEM_PROMPT, GUARDRAIL_SYSTEM_PROMPT, INTENT_SYSTEM_PROMPT, INTENTS
from rag_service import RAGService
from settings import Settings


@dataclass
class RouteResult:
    route: str
    intent: str
    translated: str
    language: str
    confidence: float
    layer_used: str
    language_hint: str
    language_hint_confidence: float
    response: str | None = None
    emotion: dict[str, Any] | None = None
    chunks: list[dict[str, Any]] = field(default_factory=list)
    guardrails: dict[str, Any] | None = None


class MentoPipeline:
    def __init__(self, settings: Settings):
        from langchain_groq import ChatGroq

        if not settings.intent_groq_api_key:
            raise RuntimeError("INTENT_GROQ_API_KEY is missing in Module 4/.env")
        if not settings.rag_groq_api_key:
            raise RuntimeError("RAG_GROQ_API_KEY is missing in Module 4/.env")

        self.settings = settings
        self.intent_llm = ChatGroq(
            api_key=settings.intent_groq_api_key,
            model=settings.intent_groq_model,
            temperature=0,
            max_tokens=220,
        )
        self.direct_llm = ChatGroq(
            api_key=settings.intent_groq_api_key,
            model=settings.intent_groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            streaming=True,
        )
        self.rag_llm = ChatGroq(
            api_key=settings.rag_groq_api_key,
            model=settings.rag_groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            streaming=True,
        )
        self.guardrail_llm = ChatGroq(
            api_key=settings.rag_groq_api_key,
            model=settings.rag_groq_model,
            temperature=0,
            max_tokens=220,
        )
        self.crisis_detector = CrisisDetector()
        self.language_detector = LanguageDetector(settings)
        self.emotion_detector = EmotionDetector(settings)
        self.rag = RAGService(settings, self.rag_llm)
        self.histories: dict[str, list[HumanMessage | AIMessage]] = {}
        self.last_mental_health_topic: dict[str, str] = {}

    def get_history(self, session_id: str) -> list[HumanMessage | AIMessage]:
        return self.histories.setdefault(session_id, [])

    def clear_history(self, session_id: str) -> None:
        self.histories.pop(session_id, None)

    def analyze_message(self, text: str, language_hint: str) -> dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                ("human", 'Message: "{message}" | hint: "{hint}"'),
            ]
        )
        chain = prompt | self.intent_llm

        last_error: Exception | None = None
        for _ in range(3):
            try:
                raw = chain.invoke({"message": text, "hint": language_hint}).content
                parsed = self._parse_json(raw)
                intent = parsed.get("intent")
                if intent not in INTENTS:
                    raise ValueError(f"Invalid intent: {intent}")
                return {
                    "intent": intent,
                    "translated": str(parsed.get("translated") or text),
                    "language": str(parsed.get("language") or language_hint),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "layer_used": "Layer 2 (LangChain Groq)",
                }
            except Exception as exc:
                last_error = exc

        return {
            "intent": "out_of_scope",
            "translated": text,
            "language": language_hint,
            "confidence": 0.0,
            "layer_used": f"Layer 2 fallback ({type(last_error).__name__ if last_error else 'unknown'})",
        }

    def classify(self, user_text: str, session_id: str | None = None) -> RouteResult:
        if self.crisis_detector.is_crisis(user_text):
            return RouteResult(
                route="crisis",
                intent="crisis",
                translated=user_text,
                language="en",
                confidence=1.0,
                layer_used="Crisis keyword guard",
                language_hint="en",
                language_hint_confidence=0.0,
                response=CRISIS_RESPONSE,
            )

        lang = self.language_detector.detect(user_text)

        # Module 1 provides only a fast language hint. The intent Groq call must
        # always verify/correct that language before downstream routing.
        llm_result = self.analyze_message(user_text, lang.language)
        if is_system_identity_question(user_text) or is_system_identity_question(llm_result["translated"]):
            llm_result["intent"] = "system_identity"
            llm_result["confidence"] = max(float(llm_result.get("confidence", 0.0)), 0.99)
        elif self._is_contextual_mental_health_followup(user_text, llm_result, session_id):
            previous_topic = self.last_mental_health_topic.get(session_id or "", "")
            llm_result["intent"] = "asking_mental_health_question"
            llm_result["translated"] = (
                "The user wants to continue discussing this mental health topic: "
                f"{previous_topic}"
            )
            llm_result["confidence"] = max(float(llm_result.get("confidence", 0.0)), 0.95)
            llm_result["layer_used"] = f"{llm_result['layer_used']} + Conversation follow-up"

        intent = llm_result["intent"]
        route = "rag_pipeline" if intent == "asking_mental_health_question" else intent
        if intent in {"greeting", "goodbye", "gratitude", "system_identity"}:
            route = "direct"
        elif intent == "out_of_scope":
            route = "out_of_scope"

        return RouteResult(
            route=route,
            intent=intent,
            translated=llm_result["translated"],
            language=llm_result["language"],
            confidence=llm_result["confidence"],
            layer_used=llm_result["layer_used"],
            language_hint=lang.language,
            language_hint_confidence=lang.confidence,
        )

    def stream(self, user_text: str, session_id: str) -> Generator[dict[str, Any], None, None]:
        user_text = (user_text or "").strip()
        if not user_text:
            yield {"type": "error", "message": "Please enter a message."}
            return

        route = self.classify(user_text, session_id)
        yield {"type": "metadata", "data": asdict(route)}

        if route.route == "crisis":
            yield from self._stream_static(route.response or CRISIS_RESPONSE)
            yield {"type": "done", "data": asdict(route)}
            return

        if route.confidence < 0.5:
            response = "I want to make sure I understand you correctly. Could you rephrase that?"
            route.route = "clarification"
            route.response = response
            yield from self._stream_static(response)
            yield {"type": "done", "data": asdict(route)}
            return

        if route.route in {"direct", "out_of_scope"}:
            response = yield from self._stream_direct_response(route, user_text)
            route.response = response
            self.last_mental_health_topic[session_id] = route.translated
            self._remember(session_id, user_text, response)
            yield {"type": "done", "data": asdict(route)}
            return

        if route.route == "rag_pipeline":
            response = yield from self._stream_rag(route, user_text, session_id)
            guarded = self.apply_guardrails(user_text, response, route.language)
            route.guardrails = guarded
            if guarded.get("is_hallucinated") and guarded.get("revised_response"):
                replacement = str(guarded["revised_response"])
                yield {"type": "replace", "text": replacement}
                response = replacement
            route.response = response
            self._remember(session_id, user_text, response)
            yield {"type": "done", "data": asdict(route)}
            return

        response = "I'm specialized in mental health support. Is there anything related to your emotional wellbeing I can help with?"
        route.response = response
        yield from self._stream_static(response)
        yield {"type": "done", "data": asdict(route)}

    def _stream_rag(self, route: RouteResult, user_text: str, session_id: str) -> Generator[dict[str, Any], None, str]:
        try:
            emotion = self.emotion_detector.detect(route.translated)
        except Exception as exc:
            emotion = None
            route.emotion = {
                "emotion": "unknown",
                "confidence": 0.0,
                "scores": {},
                "high_distress": False,
                "error": self._public_runtime_issue(exc),
            }
            yield {"type": "metadata", "data": {"emotion": route.emotion}}
            yield {"type": "notice", "message": route.emotion["error"]}

        if emotion is None:
            answer = yield from self._stream_resource_fallback(route, user_text, session_id)
            return answer

        route.emotion = asdict(emotion)
        yield {"type": "metadata", "data": {"emotion": route.emotion}}

        try:
            chunks = self.rag.retrieve_chunks(route.translated, self.settings.retriever_k)
        except Exception as exc:
            route.chunks = []
            issue = self._public_runtime_issue(exc)
            yield {"type": "metadata", "data": {"chunks": route.chunks, "rag_error": issue}}
            yield {"type": "notice", "message": issue}
            answer = yield from self._stream_resource_fallback(route, user_text, session_id)
            return answer

        route.chunks = [
            {
                "content": chunk.content,
                "source": chunk.source,
                "score": chunk.score,
                "metadata": chunk.metadata or {},
            }
            for chunk in chunks
        ]
        yield {"type": "metadata", "data": {"chunks": route.chunks}}

        chain_input = {
            "input": route.translated,
            "emotion": emotion.emotion,
            "distress_level": "HIGH" if emotion.high_distress else "NORMAL",
            "verified_language": route.language,
            "chat_history": self.get_history(session_id),
        }

        answer = ""
        try:
            for chunk in self.rag.chain.stream(chain_input, config={"metadata": {"session_id": session_id, "system": "Mento"}}):
                token = self._extract_stream_text(chunk)
                if token:
                    answer += token
                    yield {"type": "token", "text": token}
        except Exception as exc:
            issue = self._public_runtime_issue(exc)
            yield {"type": "notice", "message": issue}
            answer = yield from self._stream_resource_fallback(route, user_text, session_id)
            return answer

        if not answer.strip():
            answer = "I hear you, and I am glad you reached out. Try one small grounding step now, such as slowing your breathing, and consider speaking with someone you trust or a mental health professional for support."
            yield from self._stream_static(answer)

        return answer.strip()

    def _stream_resource_fallback(self, route: RouteResult, user_text: str, session_id: str) -> Generator[dict[str, Any], None, str]:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are Mento, a compassionate mental health support assistant. "
                        "The local RAG/emotion resources are temporarily unavailable, so answer without retrieved context. "
                        "Do not mention internal errors. Respond in the verified language. Keep it supportive and concise."
                    )
                ),
                (
                    "human",
                    "Verified language: {language}\nCleaned English message: {translated}\nOriginal message: {original}",
                ),
            ]
        )
        chain = prompt | self.rag_llm
        answer = ""
        try:
            for chunk in chain.stream(
                {
                    "language": route.language,
                    "translated": route.translated,
                    "original": user_text,
                },
                config={"metadata": {"session_id": session_id, "route": "resource_fallback", "system": "Mento"}},
            ):
                token = self._extract_stream_text(chunk)
                if token:
                    answer += token
                    yield {"type": "token", "text": token}
        except Exception:
            answer = self._localized_support_fallback(route.language)
            yield from self._stream_static(answer)
        return answer.strip()

    def _stream_direct_response(self, route: RouteResult, user_text: str) -> Generator[dict[str, Any], None, str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=DIRECT_RESPONSE_SYSTEM_PROMPT),
                (
                    "human",
                    "Intent: {intent}\nVerified language: {language}\nUser message: {message}",
                ),
            ]
        )
        chain = prompt | self.direct_llm
        answer = ""
        for chunk in chain.stream(
            {
                "intent": route.intent,
                "language": route.language,
                "message": user_text,
            },
            config={"metadata": {"route": route.route, "system": "Mento"}},
        ):
            token = self._extract_stream_text(chunk)
            if token:
                answer += token
                yield {"type": "token", "text": token}
        if not answer.strip():
            answer = self._direct_fallback(route)
            yield from self._stream_static(answer)
        return answer.strip()

    def apply_guardrails(self, user_text: str, response: str, language: str) -> dict[str, Any]:
        heuristic = heuristic_guardrail(user_text, response)
        if heuristic["is_hallucinated"]:
            return heuristic

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
                (
                    "human",
                    "Language: {language}\nUser text: {user_text}\nGenerated response: {response}",
                ),
            ]
        )
        try:
            raw = (prompt | self.guardrail_llm).invoke(
                {"language": language, "user_text": user_text, "response": response}
            ).content
            parsed = self._parse_json(raw)
            safe = bool(parsed.get("safe", True))
            return {
                "is_hallucinated": not safe,
                "flags": [] if safe else [str(parsed.get("reason", "LLM guardrail flagged response"))],
                "revised_response": parsed.get("revised_response") if not safe else None,
            }
        except Exception:
            return heuristic

    def _remember(self, session_id: str, user_text: str, response: str) -> None:
        history = self.get_history(session_id)
        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=response))
        if len(history) > 12:
            del history[: len(history) - 12]

    def _direct_fallback(self, route: RouteResult) -> str:
        if route.route == "out_of_scope":
            return "I'm specialized in mental health support. Is there anything related to your emotional wellbeing I can help with?"
        if route.intent == "system_identity":
            return "My name is Mento. I'm here to support your emotional wellbeing."
        if route.intent == "gratitude":
            return "You're very welcome. I'm here whenever you need support."
        if route.intent == "goodbye":
            return "Take care of yourself. Support is here whenever you need it."
        return "Hello, I'm Mento. I'm here to support your emotional wellbeing."

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = re.sub(r"```json|```", "", raw or "").strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)
        return json.loads(cleaned)

    @staticmethod
    def _extract_stream_text(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            if "answer" in chunk:
                value = chunk["answer"]
                return value if isinstance(value, str) else ""
            if "text" in chunk:
                return str(chunk["text"])
            return ""
        content = getattr(chunk, "content", None)
        return content if isinstance(content, str) else ""

    @staticmethod
    def _public_runtime_issue(exc: Exception) -> str:
        message = str(exc).lower()
        if isinstance(exc, MemoryError) or "paging file is too small" in message or "os error 1455" in message:
            return "Local memory is too low to load the RAG resources right now, so I am using a safe fallback response."
        if "network error" in message or "connection" in message:
            return "A RAG network call was unavailable, so I am using a safe fallback response."
        return "A local RAG resource was unavailable, so I am using a safe fallback response."

    @staticmethod
    def _localized_support_fallback(language: str) -> str:
        if language == "ar":
            return "أفهم ما تشعر به، وأنا هنا لدعمك. جرّب أخذ نفس بطيء الآن، وإذا كان الشعور شديداً فتحدث مع شخص تثق به أو مختص في الصحة النفسية."
        return "I hear you, and I am here with you. Try one small grounding step now, like slowing your breathing, and consider talking with someone you trust or a mental health professional."

    def _is_contextual_mental_health_followup(
        self,
        user_text: str,
        llm_result: dict[str, Any],
        session_id: str | None,
    ) -> bool:
        if not session_id or session_id not in self.last_mental_health_topic:
            return False
        if llm_result.get("intent") == "asking_mental_health_question":
            return False
        return is_followup_affirmation(user_text)

    @staticmethod
    def _stream_static(text: str) -> Generator[dict[str, Any], None, None]:
        words = text.split(" ")
        for index, word in enumerate(words):
            suffix = " " if index < len(words) - 1 else ""
            yield {"type": "token", "text": word + suffix}
