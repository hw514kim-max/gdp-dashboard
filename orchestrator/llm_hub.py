"""Claude가 GPT / Gemini에게 일을 시키기 위한 얇은 호출 모듈.

사용 예:
    python orchestrator/llm_hub.py gpt "질문"
    python orchestrator/llm_hub.py gemini "질문"
    python orchestrator/llm_hub.py both "질문"     # 두 모델에 동시에 물어 비교

모델명은 환경변수로 교체 가능: OPENAI_MODEL, GEMINI_MODEL
키는 환경변수 OPENAI_API_KEY / GEMINI_API_KEY (클라우드에서는 프록시가 주입하므로 더미값 가능).
"""
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor

import requests

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
TIMEOUT = 120


def ask_gpt(prompt: str, system: str = "", model: str = OPENAI_MODEL) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy')}"},
        json={"model": model, "messages": messages},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_gemini(prompt: str, system: str = "", model: str = GEMINI_MODEL) -> str:
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ.get("GEMINI_API_KEY", "dummy")},
        json=body,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_both(prompt: str, system: str = "") -> dict:
    """두 모델에 병렬로 질문. 한쪽이 실패해도 다른 쪽 결과는 반환."""
    def safe(fn):
        try:
            return fn(prompt, system)
        except Exception as e:  # noqa: BLE001
            return f"[오류] {e}"
    with ThreadPoolExecutor(2) as ex:
        g, m = ex.submit(safe, ask_gpt), ex.submit(safe, ask_gemini)
        return {"gpt": g.result(), "gemini": m.result()}


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("gpt", "gemini", "both"):
        sys.exit(__doc__)
    who, q = sys.argv[1], " ".join(sys.argv[2:])
    if who == "gpt":
        print(ask_gpt(q))
    elif who == "gemini":
        print(ask_gemini(q))
    else:
        print(json.dumps(ask_both(q), ensure_ascii=False, indent=2))
