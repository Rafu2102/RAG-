# -*- coding: utf-8 -*-
"""
llm/gemini_client.py — 統一 Gemini API 呼叫封裝 (Google Cloud Vertex AI ADC 企業版)
==================================================================================
全面停用 AI Studio 金鑰，改用 Google Cloud 原生 Vertex AI 端點。
使用 Google 應用程式預設憑證 (Application Default Credentials, ADC) 無密鑰安全授權。

⚠️ Gemini 3.x REST API 參數規範（依據官方文件 2026-05）：
   - temperature / top_p / top_k 不建議使用，已從 payload 移除
   - thinkingConfig.thinkingLevel 值為小寫字串，例如 'high'、'medium'、'low'，Gemini 3.5 全系模型（含 Flash 與 Pro）均已原生支援此思考配置
"""

import os
import asyncio
import json
import logging
import random
import time

import httpx
import requests
import google.auth
from google.auth.transport.requests import Request as SyncRequest

import config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 🔑 Google Cloud ADC (應用程式預設憑證) 憑證與 Token 緩存引擎
# ═══════════════════════════════════════════════════════════════
_gcp_credentials = None
_gcp_project_id = None
_token_expiry = 0.0

def get_gcp_auth_info() -> tuple[str, str]:
    """自動且免密鑰獲取本地或雲端環境的 Google Cloud ADC Access Token 及 Project ID。
    內置高性能 Token 緩存機制，避免每筆 API 呼叫頻繁 refresh。
    """
    global _gcp_credentials, _gcp_project_id, _token_expiry
    now = time.time()
    
    # 若憑證尚未加載，或 Token 即將於 60 秒內過期，則進行刷新
    if _gcp_credentials is None or now >= _token_expiry - 60.0:
        logger.info("🔑 正在讀取並刷新 Google Cloud 應用程式預設憑證 (ADC)...")
        _gcp_credentials, _gcp_project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        _gcp_credentials.refresh(SyncRequest())
        # 強制設定為未來 3500 秒 (約 58 分鐘)，避免 google.auth 傳回 UTC naive datetime 導致 timestamp 換算錯誤而無限刷新
        _token_expiry = now + 3500.0
        logger.info(f"✨ GCP ADC 憑證加載成功！當前 Project ID: {_gcp_project_id}")
        
    return _gcp_credentials.token, _gcp_project_id

def _build_vertex_request(model_name: str, payload: dict) -> tuple[str, dict, dict]:
    """構建符合 Google Cloud Vertex AI REST 規範的 (url, headers, payload)。"""
    token, project_id = get_gcp_auth_info()
    
    url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model_name}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    return url, headers, payload

# ═══════════════════════════════════════════════════════════════
# 📡 全域非同步 HTTP 用戶端單例，支援連接池複用，避免頻繁建連開銷
# ═══════════════════════════════════════════════════════════════
_async_client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    timeout=None,  # 讓每次呼叫依據 model 指定的逾時常數處理
)


# ═══════════════════════════════════════════════════════════════
# 🛡️ 帶有隨機抖動與指數退避之重試引擎 (Sync & Async)
# ═══════════════════════════════════════════════════════════════

async def _a_post_with_retry(
    model_name: str,
    payload: dict,
    timeout: float,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> httpx.Response:
    """帶有隨機抖動與指數退避的非同步 POST 請求給 Vertex AI。"""
    for attempt in range(max_retries + 1):
        try:
            url, headers, vertex_payload = _build_vertex_request(model_name, payload)
            resp = await _async_client.post(url, json=vertex_payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as he:
            status_code = he.response.status_code
            if status_code in [429, 500, 502, 503] and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(
                    f"⚠️ Vertex AI API 回傳 {status_code} ，正在進行第 {attempt + 1} 次非同步重試，將等待 {delay:.2f} 秒..."
                )
                await asyncio.sleep(delay)
                continue
            if status_code == 429:
                raise GeminiQuotaError("Vertex AI API 配額耗盡 (429)") from he
            raise GeminiAPIError(f"HTTP {status_code} : {he}") from he
        except (httpx.RequestError, asyncio.TimeoutError) as re:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(
                    f"⚠️ Vertex AI API 網路異常或超時 ({type(re).__name__}) ，正在進行第 {attempt + 1} 次非同步重試，將等待 {delay:.2f} 秒..."
                )
                await asyncio.sleep(delay)
                continue
            raise GeminiAPIError(f"Vertex AI API 網路超時或連線失敗: {re}") from re
        except Exception as e:
            raise GeminiAPIError(f"Vertex AI 呼叫發生未預期錯誤: {e}") from e
    raise GeminiAPIError("重試次數超出上限")


def _post_with_retry(
    model_name: str,
    payload: dict,
    timeout: float,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> requests.Response:
    """帶有隨機抖動與指數退避的同步 POST 請求給 Vertex AI。"""
    for attempt in range(max_retries + 1):
        try:
            url, headers, vertex_payload = _build_vertex_request(model_name, payload)
            resp = requests.post(url, json=vertex_payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as he:
            status_code = he.response.status_code if he.response is not None else 500
            if status_code in [429, 500, 502, 503] and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(
                    f"⚠️ Vertex AI API 回傳 {status_code} ，正在進行第 {attempt + 1} 次同步重試，將等待 {delay:.2f} 秒..."
                )
                time.sleep(delay)
                continue
            if status_code == 429:
                raise GeminiQuotaError("Vertex AI API 配額耗盡 (429)") from he
            raise GeminiAPIError(f"HTTP {status_code} : {he}") from he
        except (requests.exceptions.RequestException, Exception) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(
                    f"⚠️ Vertex AI API 同步呼叫異常 ({type(e).__name__}) ，正在進行第 {attempt + 1} 次同步重試，將等待 {delay:.2f} 秒..."
                )
                time.sleep(delay)
                continue
            raise GeminiAPIError(f"Vertex AI API 同步網路超時或連線失敗: {e}") from e
    raise GeminiAPIError("重試次數超出上限")


# ═══════════════════════════════════════════════════════════════
# 📡 核心同步呼叫封裝 (向後相容)
# ═══════════════════════════════════════════════════════════════

def call_gemini(
    prompt: str | list,
    *,
    model: str = "pro",
    thinking: str = "medium",
    max_tokens: int | None = None,
    tools: list | None = None,
    timeout: float | None = None,
) -> str:
    """呼叫 Vertex AI Gemini API 並回傳純文字結果（已加入同步指數退避重試，支援 ADC）。"""
    model_name, default_timeout, default_tokens = _resolve_model(model, thinking)

    contents = prompt if isinstance(prompt, list) else {"role": "user", "parts": [{"text": prompt}]}
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens or default_tokens,
            "thinkingConfig": {"thinkingLevel": _resolve_thinking_level(thinking)},
        },
    }

    if tools:
        payload["tools"] = tools

    resp = _post_with_retry(model_name, payload, timeout=timeout or default_timeout)
    return _extract_text(resp.json())


def call_gemini_json(
    prompt: str,
    schema: dict,
    *,
    model: str = "fast",
    thinking: str = "low",
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> dict:
    """呼叫 Vertex AI Gemini API 並回傳結構化 JSON 結果（已加入同步指數退避重試，支援 ADC）。"""
    model_name, default_timeout, default_tokens = _resolve_model(model, thinking)

    payload = {
        "contents": {"role": "user", "parts": [{"text": prompt}]},
        "generationConfig": {
            "maxOutputTokens": max_tokens or default_tokens,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "thinkingConfig": {"thinkingLevel": _resolve_thinking_level(thinking)},
        },
    }

    resp = _post_with_retry(model_name, payload, timeout=timeout or default_timeout)
    try:
        text = _extract_text(resp.json())
        return json.loads(text)
    except json.JSONDecodeError as je:
        raise GeminiAPIError(f"JSON 解析失敗: {je}") from je


def call_gemini_with_fallback(
    prompt: str,
    *,
    thinking: str = "medium",
    max_tokens: int | None = None,
    timeout: float | None = None,
    tools: list | None = None,
) -> str:
    """先嘗試 Pro 設定，若遭遇 429 或其他大 context 思考截斷錯誤，則自動進行兩階段降級重試。"""
    thinking_level = str(thinking).upper()
    if thinking_level not in ["MINIMAL", "LOW", "MEDIUM", "HIGH"]:
        thinking_level = "MEDIUM"

    try:
        logger.info(f"✨ 嘗試同步 Pro 呼叫，Thinking Level: {thinking_level}")
        return call_gemini(
            prompt, model="pro", thinking=thinking_level,
            max_tokens=max_tokens, timeout=timeout, tools=tools,
        )
    except GeminiQuotaError as eq:
        logger.warning(f"⚠️ Vertex AI Pro 配額耗盡 (429) ，自動降級為 Fast 呼叫：{eq}")
        return call_gemini(
            prompt, model="fast", thinking="MEDIUM",
            max_tokens=max_tokens, timeout=timeout, tools=tools,
        )
    except GeminiAPIError as ee:
        if thinking_level == "HIGH":
            logger.warning(f"⚠️ 同步 HIGH 思考等級呼叫失敗 ({ee})，自動原地降級為 MEDIUM 思考等級重試...")
            try:
                return call_gemini(
                    prompt, model="pro", thinking="MEDIUM",
                    max_tokens=max_tokens, timeout=timeout, tools=tools,
                )
            except GeminiAPIError as ee2:
                logger.warning(f"⚠️ 同步原地降級 MEDIUM 仍失敗：{ee2}，進一步降級為 Fast 呼叫...")
        else:
            logger.warning(f"⚠️ 同步 Pro 呼叫失敗 ({ee})，自動降級為 Fast 呼叫...")
            
        return call_gemini(
            prompt, model="fast", thinking="MEDIUM",
            max_tokens=max_tokens, timeout=timeout, tools=tools,
        )


# ═══════════════════════════════════════════════════════════════
# 📡 核心非同步呼叫封裝 (高效 Non-blocking)
# ═══════════════════════════════════════════════════════════════

async def a_call_gemini(
    prompt: str | list,
    *,
    model: str = "pro",
    thinking: str = "medium",
    max_tokens: int | None = None,
    tools: list | None = None,
    timeout: float | None = None,
) -> str:
    """非同步呼叫 Vertex AI Gemini API 並回傳純文字結果（支援指數退避重試與連接池）。"""
    model_name, default_timeout, default_tokens = _resolve_model(model, thinking)

    contents = prompt if isinstance(prompt, list) else {"role": "user", "parts": [{"text": prompt}]}
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens or default_tokens,
            "thinkingConfig": {"thinkingLevel": _resolve_thinking_level(thinking)},
        },
    }

    if tools:
        payload["tools"] = tools

    resp = await _a_post_with_retry(model_name, payload, timeout=timeout or default_timeout)
    return _extract_text(resp.json())


async def a_call_gemini_json(
    prompt: str,
    schema: dict,
    *,
    model: str = "fast",
    thinking: str = "low",
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> dict:
    """非同步呼叫 Vertex AI Gemini API 並回傳結構化 JSON 結果（支援指數退避重試與連接池）。"""
    model_name, default_timeout, default_tokens = _resolve_model(model, thinking)

    payload = {
        "contents": {"role": "user", "parts": [{"text": prompt}]},
        "generationConfig": {
            "maxOutputTokens": max_tokens or default_tokens,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "thinkingConfig": {"thinkingLevel": _resolve_thinking_level(thinking)},
        },
    }

    resp = await _a_post_with_retry(model_name, payload, timeout=timeout or default_timeout)
    try:
        text = _extract_text(resp.json())
        return json.loads(text)
    except json.JSONDecodeError as je:
        raise GeminiAPIError(f"JSON 解析失敗: {je}") from je


async def a_call_gemini_with_fallback(
    prompt: str,
    *,
    thinking: str = "medium",
    max_tokens: int | None = None,
    timeout: float | None = None,
    tools: list | None = None,
) -> str:
    """非同步先嘗試 Pro 設定，若遭遇 429 或其他大 context 思考截斷錯誤，則自動進行兩階段降級重試。"""
    thinking_level = str(thinking).upper()
    if thinking_level not in ["MINIMAL", "LOW", "MEDIUM", "HIGH"]:
        thinking_level = "MEDIUM"

    try:
        logger.info(f"✨ 嘗試非同步 Pro 呼叫，Thinking Level: {thinking_level}")
        return await a_call_gemini(
            prompt, model="pro", thinking=thinking_level,
            max_tokens=max_tokens, timeout=timeout, tools=tools,
        )
    except GeminiQuotaError as eq:
        logger.warning(f"⚠️ Vertex AI Pro 配額耗盡 (429) ，自動降級為 Fast 呼叫：{eq}")
        return await a_call_gemini(
            prompt, model="fast", thinking="MEDIUM",
            max_tokens=max_tokens, timeout=timeout, tools=tools,
        )
    except GeminiAPIError as ee:
        if thinking_level == "HIGH":
            logger.warning(f"⚠️ 非同步 HIGH 思考等級呼叫失敗 ({ee})，自動原地降級為 MEDIUM 思考等級重試...")
            try:
                return await a_call_gemini(
                    prompt, model="pro", thinking="MEDIUM",
                    max_tokens=max_tokens, timeout=timeout, tools=tools,
                )
            except GeminiAPIError as ee2:
                logger.warning(f"⚠️ 非同步原地降級 MEDIUM 仍失敗：{ee2}，進一步降級為 Fast 呼叫...")
        else:
            logger.warning(f"⚠️ 非同步 Pro 呼叫失敗 ({ee})，自動降級為 Fast 呼叫...")
            
        return await a_call_gemini(
            prompt, model="fast", thinking="MEDIUM",
            max_tokens=max_tokens, timeout=timeout, tools=tools,
        )


async def a_call_gemini_raw(
    model_name: str,
    payload: dict,
    timeout: float | None = None,
) -> str:
    """底層通用非同步 API 呼叫，接受完整 payload，自動載入 GCP ADC 認證，並回傳純文字結果。
    這可供 vision OCR、多模態、或具有自訂 generationConfig 的元件使用。
    """
    if timeout is None:
        timeout = config.GEMINI_PRO_TIMEOUT
    
    resp = await _a_post_with_retry(model_name, payload, timeout=timeout)
    return _extract_text(resp.json())


# ═══════════════════════════════════════════════════════════════
# 🔧 內部工具
# ═══════════════════════════════════════════════════════════════

# (已移除 _VALID_THINKING_LEVELS 與 _normalize_thinking_level)


def _resolve_thinking_level(thinking: str) -> str:
    """將抽象思考等級字串轉換為符合 Vertex AI v1 REST 規範的 thinkingLevel 字串（大寫）。"""
    if not thinking:
        return "MEDIUM"
    
    val = str(thinking).upper()
    valid_levels = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
    return val if val in valid_levels else "MEDIUM"


def _resolve_model(model: str, thinking: str = "medium") -> tuple[str, float, int]:
    """根據 model 名稱與思考等級回傳 (model_name, timeout, max_tokens)。"""

    if model in ["gemini-3.1-pro", "gemini-3.1-pro-preview"]:
        return "gemini-3.1-pro-preview", config.GEMINI_PRO_TIMEOUT, config.GEMINI_PRO_MAX_TOKENS
        
    # 依使用者最新指示：一般 RAG 回答（包含 pro 與 fast 配置）均統一使用 gemini-3.5-flash
    if model == "fast":
        return "gemini-3.5-flash", config.GEMINI_FLASH_TIMEOUT, config.GEMINI_FLASH_MAX_TOKENS
    
    # 預設 pro 映射：同樣指向 gemini-3.5-flash，但保留 pro 原本較寬鬆的超時與最大 token 限制
    return "gemini-3.5-flash", config.GEMINI_PRO_TIMEOUT, config.GEMINI_PRO_MAX_TOKENS


def _extract_text(response_json: dict) -> str:
    """從 Gemini API 回應中提取文字內容，並加入 finishReason 診斷與防禦性錯誤處理。"""
    candidates = response_json.get("candidates", [])
    if not candidates:
        raise GeminiAPIError("Gemini 未回傳任何 candidate")
    
    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "STOP")
    
    # 🚨 處理安全過濾
    if finish_reason == "SAFETY":
        raise GeminiAPIError("回答因觸發安全過濾機制而被截斷 (SAFETY)")
        
    content = candidate.get("content", {})
    parts = content.get("parts", [])
    
    # 🚨 處理 MAX_TOKENS 截斷或 parts 為空的情況
    if not parts:
        if finish_reason == "MAX_TOKENS":
            raise GeminiAPIError("回答因輸出 Token 數達到上限而被截斷 (MAX_TOKENS)")
        raise GeminiAPIError(f"Gemini 回應中 content 未包含 parts (finishReason: {finish_reason})")
    
    # 提取所有非 thought (或 thought != True) 的 text
    text_parts = []
    for part in parts:
        is_thought = part.get("thought", False)
        if not is_thought:
            t = part.get("text", "")
            if t:
                text_parts.append(t)
                
    if not text_parts:
        # Fallback 拿第一個 part 的 text 進行最大程度防禦
        t = parts[0].get("text", "")
        if t:
            return t.strip()
        raise GeminiAPIError("Gemini 回應 parts 中未包含有效的 text")
        
    return "".join(text_parts).strip()


# ═══════════════════════════════════════════════════════════════
# ⚠️ 例外類別
# ═══════════════════════════════════════════════════════════════

class GeminiAPIError(Exception):
    """Gemini API 呼叫失敗的通用例外。"""
    pass


class GeminiQuotaError(GeminiAPIError):
    """Gemini API 配額耗盡 (HTTP 429)。"""
    pass
