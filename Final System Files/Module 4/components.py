from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from settings import Settings


LANGUAGE_MAP = {
    0: "ar",
    1: "bg",
    2: "de",
    3: "el",
    4: "en",
    5: "es",
    6: "fr",
    7: "hi",
    8: "it",
    9: "ja",
    10: "nl",
    11: "pl",
    12: "pt",
    13: "ru",
    14: "sw",
    15: "th",
    16: "tr",
    17: "ur",
    18: "vi",
    19: "zh",
}

ID2EMOTION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

CRISIS_KEYWORDS = {
    "en": [
        "kill myself",
        "end my life",
        "want to die",
        "suicide",
        "suicidal",
        "hurt myself",
        "self harm",
        "self-harm",
        "cutting myself",
        "want to hurt myself",
        "no reason to live",
        "better off dead",
        "overdose",
        "jump off",
        "hang myself",
    ],
    "ar": ["أريد الموت", "انتحار", "أؤذي نفسي", "أقتل نفسي", "لا أريد العيش"],
    "fr": ["me suicider", "me tuer", "mettre fin à ma vie"],
    "de": ["mich umbringen", "suizid", "selbstmord"],
    "es": ["suicidarme", "matarme", "quiero morir"],
}

CRISIS_RESPONSE = (
    "I'm really concerned about what you've shared, and I want you to know you're not alone. "
    "If you might hurt yourself or feel unable to stay safe, please contact local emergency "
    "services now or reach a crisis helpline immediately. In the US or Canada, call or text "
    "988. If you are elsewhere, contact your local emergency number or use "
    "https://www.iasp.info/resources/Crisis_Centres/ to find support. Please also tell someone "
    "near you what is happening right now."
)


@dataclass
class LanguageResult:
    language: str
    confidence: float


@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    scores: dict[str, float]
    high_distress: bool
    source: str = "local_model"


class CrisisDetector:
    def is_crisis(self, text: str) -> bool:
        normalized = (text or "").lower()
        return any(keyword.lower() in normalized for words in CRISIS_KEYWORDS.values() for keyword in words)


class LanguageDetector:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def vectorizer(self) -> Any:
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return joblib.load(self._first_existing(["tfidf_vectorizer.pkl", "ang_vectorizer.pkl"]))

    @cached_property
    def classifier(self) -> Any:
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return joblib.load(self._first_existing(["language_classifier.pkl", "ml_lang_model.pkl"]))

    def _first_existing(self, names: list[str]) -> Path:
        for name in names:
            path = self.settings.module1_models_dir / name
            if path.exists():
                return path
        choices = ", ".join(names)
        raise FileNotFoundError(f"Could not find Module 1 model file. Tried: {choices}")

    def detect(self, text: str) -> LanguageResult:
        if not text or not isinstance(text, str):
            return LanguageResult("en", 0.0)

        features = self.vectorizer.transform([text.strip().lower()])
        prediction = self.classifier.predict(features)[0]
        language = LANGUAGE_MAP.get(int(prediction), str(prediction)) if isinstance(prediction, (int, np.integer)) else str(prediction)

        confidence = 0.0
        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

        return LanguageResult(language=language, confidence=confidence)


class EmotionDetector:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(self.settings.module2_model_dir))

    @cached_property
    def model(self) -> Any:
        import torch
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(str(self.settings.module2_model_dir))
        model.to(self.device)
        model.eval()
        return model

    @cached_property
    def device(self) -> Any:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, text: str, max_len: int = 128) -> EmotionResult:
        if not text or not isinstance(text, str):
            return EmotionResult("neutral", 0.0, {}, False, "empty")

        if not self.settings.use_local_emotion_model or not has_available_pagefile(
            self.settings.min_available_pagefile_mb
        ):
            return heuristic_emotion(text)

        import torch

        encoded = self.tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()

        emotion_id = int(np.argmax(probabilities))
        emotion = ID2EMOTION.get(emotion_id, "unknown")
        confidence = float(probabilities[emotion_id])
        scores = {ID2EMOTION.get(i, f"label_{i}"): float(score) for i, score in enumerate(probabilities)}
        high_distress = emotion in {"sadness", "anger", "fear"} and confidence > 0.6

        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            scores=scores,
            high_distress=high_distress,
            source="local_model",
        )


def available_pagefile_mb() -> int | None:
    if sys.platform != "win32":
        return None

    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPageFile / (1024 * 1024))
    except Exception:
        return None


def has_available_pagefile(min_mb: int) -> bool:
    available = available_pagefile_mb()
    if available is None:
        return True
    return available >= min_mb


def heuristic_emotion(text: str) -> EmotionResult:
    lowered = (text or "").lower()
    buckets = {
        "sadness": [
            "sad",
            "lonely",
            "alone",
            "cry",
            "hopeless",
            "grief",
            "empty",
            "وحدة",
            "وحيد",
            "حزين",
            "حزن",
            "ببكي",
        ],
        "fear": [
            "anxious",
            "anxiety",
            "afraid",
            "fear",
            "panic",
            "worried",
            "قلق",
            "خائف",
            "خوف",
            "هلع",
        ],
        "anger": [
            "angry",
            "mad",
            "furious",
            "rage",
            "غاضب",
            "غضب",
            "متضايق",
        ],
        "joy": ["happy", "good", "better", "فرح", "سعيد"],
        "love": ["love", "loved", "care", "حب"],
        "surprise": ["surprised", "shocked", "مندهش"],
    }

    for emotion, keywords in buckets.items():
        if any(keyword in lowered for keyword in keywords):
            confidence = 0.72
            scores = {label: 0.05 for label in ID2EMOTION.values()}
            scores[emotion] = confidence
            return EmotionResult(
                emotion=emotion,
                confidence=confidence,
                scores=scores,
                high_distress=emotion in {"sadness", "anger", "fear"},
                source="heuristic_memory_safe",
            )

    return EmotionResult(
        emotion="sadness",
        confidence=0.45,
        scores={"sadness": 0.45},
        high_distress=False,
        source="heuristic_memory_safe",
    )


def regex_direct_intent(text: str) -> str | None:
    stripped = (text or "").strip().lower()
    if len(stripped.split()) > 4:
        return None

    patterns = {
        "greeting": [
            r"\bhi\b",
            r"\bhello\b",
            r"\bhey\b",
            r"\bgood morning\b",
            r"\bgood evening\b",
            r"\bgood afternoon\b",
            r"\bhowdy\b",
            r"\bgreetings\b",
        ],
        "goodbye": [
            r"\bbye\b",
            r"\bgoodbye\b",
            r"\bfarewell\b",
            r"\bsee you\b",
            r"\btake care\b",
            r"\bgoodnight\b",
        ],
        "gratitude": [
            r"\bthank you\b",
            r"\bthanks\b",
            r"\bthank u\b",
            r"\bgrateful\b",
            r"\bmany thanks\b",
        ],
    }

    for intent, intent_patterns in patterns.items():
        if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in intent_patterns):
            return intent
    return None


def is_system_identity_question(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return False

    identity_patterns = [
        r"\bwhat'?s your name\b",
        r"\bwhat is your name\b",
        r"\bwho are you\b",
        r"\byour name\b",
        r"\bsystem name\b",
        r"\bاسمك\b",
        r"\bما اسمك\b",
        r"\bمين انت\b",
        r"\bمن أنت\b",
    ]
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in identity_patterns)


def is_followup_affirmation(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return False

    if len(normalized.split()) > 5:
        return False

    affirmative_patterns = [
        r"^yes$",
        r"^yeah$",
        r"^yep$",
        r"^sure$",
        r"^please$",
        r"^ok(?:ay)?$",
        r"^continue$",
        r"^yes please$",
        r"^i want$",
        r"^i want to$",
        r"^نعم$",
        r"^نعم اريد$",
        r"^نعم أريد$",
        r"^اريد$",
        r"^أريد$",
        r"^ايوا$",
        r"^أيوه$",
        r"^ايوه$",
        r"^اجل$",
        r"^أجل$",
        r"^تمام$",
        r"^اه$",
        r"^آه$",
    ]
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in affirmative_patterns)


def heuristic_guardrail(user_text: str, generated_response: str) -> dict[str, Any]:
    flags: list[str] = []
    false_attributions = [
        ("you are depressed", "depression"),
        ("you have anxiety", "anxiety"),
        ("you are suicidal", "suicide"),
    ]
    lower_user = (user_text or "").lower()
    lower_response = (generated_response or "").lower()

    for phrase, condition in false_attributions:
        if phrase in lower_response and condition not in lower_user:
            flags.append(f"Potential false attribution: {phrase}")

    for claim in ("you are cured", "you will be fine in", "this will definitely solve"):
        if claim in lower_response:
            flags.append(f"Suspicious medical claim: {claim}")

    return {"is_hallucinated": bool(flags), "flags": flags}
