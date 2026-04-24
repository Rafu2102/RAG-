# -*- coding: utf-8 -*-
"""
llm/gemini_client.py — 統一 Gemini API 呼叫封裝
==================================================
消除散落在 10+ 個檔案中的重複 `requests.post → json → candidates[0]` 模式。
提供兩個核心函式：
  - call_gemini()       — 純文字生成
  - call_gemini_json()  — 結構化 JSON 生成（附 responseSchema）
"""

import json
import logging

import requests

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 📡 核心呼叫封裝
# ═══════════════════════════════════════════════════════════════

def call_gemini(
    prompt: str,
    *,
    model: str = "pro",
    thinking: str = "medium",
    max_tokens: int | None = None,
    tools: list | None = None,
    timeout: float | None = None,
) -> str:
    """呼叫 Gemini API 並回傳純文字結果。

    Args:
        prompt: 使用者 prompt 全文。
        model: "pro" 或 "fast"，對應 Gemini 3.1 Pro / Flash Lite。
        thinking: "minimal" | "low" | "medium" | "high"。
        max_tokens: 最大輸出 token 數，預設依 model 自動選擇。
        tools: 額外工具（如 google_search）。
        timeout: 請求 timeout 秒數，預設依 model 自動選擇。

    Returns:
        Gemini 生成的文字內容（已 strip）。

    Raises:
        GeminiQuotaError: 當 Pro 達到 429 配額上限時拋出。
        GeminiAPIError: 其他 API 錯誤。
    """
    url, default_timeout, default_tokens = _resolve_model(model)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": max_tokens or default_tokens,
            "thinkingConfig": {"thinkingLevel": thinking},
        },
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(url, json=payload, timeout=timeout or default_timeout)
        resp.raise_for_status()
        return _extract_text(resp.json())
    except requests.exceptions.HTTPError as he:
        if he.response is not None and he.response.status_code == 429:
            raise GeminiQuotaError("Gemini API 配額耗盡 (429)") from he
        raise GeminiAPIError(f"HTTP {he.response.status_code if he.response else '?'}: {he}") from he
    except Exception as e:
        raise GeminiAPIError(f"Gemini 呼叫失敗: {e}") from e


def call_gemini_json(
    prompt: str,
    schema: dict,
    *,
    model: str = "fast",
    thinking: str = "low",
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> dict:
    """呼叫 Gemini API 並回傳結構化 JSON 結果。

    Args:
        prompt: 使用者 prompt 全文。
        schema: Gemini responseSchema 定義。
        model: "pro" 或 "fast"。
        thinking: thinking level。
        max_tokens: 最大輸出 token 數。
        timeout: 請求 timeout 秒數。

    Returns:
        解析後的 dict。

    Raises:
        GeminiQuotaError: 配額耗盡。
        GeminiAPIError: 其他錯誤。
    """
    url, default_timeout, default_tokens = _resolve_model(model)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": max_tokens or default_tokens,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "thinkingConfig": {"thinkingLevel": thinking},
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout or default_timeout)
        resp.raise_for_status()
        text = _extract_text(resp.json())
        return json.loads(text)
    except requests.exceptions.HTTPError as he:
        if he.response is not None and he.response.status_code == 429:
            raise GeminiQuotaError("Gemini API 配額耗盡 (429)") from he
        raise GeminiAPIError(f"HTTP {he.response.status_code if he.response else '?'}: {he}") from he
    except json.JSONDecodeError as je:
        raise GeminiAPIError(f"JSON 解析失敗: {je}") from je
    except Exception as e:
        raise GeminiAPIError(f"Gemini 呼叫失敗: {e}") from e


def call_gemini_with_fallback(
    prompt: str,
    *,
    thinking: str = "medium",
    max_tokens: int | None = None,
    timeout: float | None = None,
    tools: list | None = None,
) -> str:
    """先嘗試 Pro，若 429 則自動降級至 Flash Lite。

    Args:
        prompt: 使用者 prompt 全文。
        thinking: Pro 層級的 thinking level。
        max_tokens: 最大輸出 token 數。
        timeout: 請求 timeout 秒數。
        tools: 額外工具。

    Returns:
        Gemini 生成的文字內容。
    """
    try:
        return call_gemini(
            prompt, model="pro", thinking=thinking,
            max_tokens=max_tokens, timeout=timeout, tools=tools,
        )
    except GeminiQuotaError:
        logger.warning("⚠️ Gemini Pro 配額耗盡 (429)，自動降級至 Flash Lite")
        return call_gemini(
            prompt, model="fast", thinking="medium",
            max_tokens=max_tokens, timeout=timeout, tools=tools,
        )


# ═══════════════════════════════════════════════════════════════
# 🔧 內部工具
# ═══════════════════════════════════════════════════════════════

def _resolve_model(model: str) -> tuple[str, float, int]:
    """根據 model 名稱回傳 (url, timeout, max_tokens)。"""
    if model == "fast":
        return config.GEMINI_FAST_API_URL, config.GEMINI_FLASH_TIMEOUT, config.GEMINI_FLASH_MAX_TOKENS
    # 預設為 pro
    return config.GEMINI_API_URL, config.GEMINI_PRO_TIMEOUT, config.GEMINI_PRO_MAX_TOKENS


def _extract_text(response_json: dict) -> str:
    """從 Gemini API 回應中提取文字內容。"""
    candidates = response_json.get("candidates", [])
    if not candidates:
        raise GeminiAPIError("Gemini 未回傳任何 candidate")
    return candidates[0]["content"]["parts"][0]["text"].strip()


# ═══════════════════════════════════════════════════════════════
# ⚠️ 例外類別
# ═══════════════════════════════════════════════════════════════

class GeminiAPIError(Exception):
    """Gemini API 呼叫失敗的通用例外。"""


class GeminiQuotaError(GeminiAPIError):
    """Gemini API 配額耗盡 (HTTP 429)。"""
