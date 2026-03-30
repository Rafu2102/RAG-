# -*- coding: utf-8 -*-
"""
dcard_search_tool.py — 校園資訊聚合搜尋引擎
=============================================
整合三大引擎，為金門大學學生提供全方位的即時校園資訊：

📰 來源 1 — NQU 官網爬蟲
   爬取學校官方「校內公告」與「學術焦點動態」，
   提取最新的招生簡章、行政公告、活動訊息等。

💬 來源 2 — Dcard 金大版搜尋
   透過 OpenRouter (Perplexity Sonar Pro)
   搜尋 Dcard 金門大學板上的教授評價與課程推薦。

🤖 總結引擎 — Gemini 3.1 Pro
   將兩個來源的原始資料餵給 Gemini，由 AI 進行智慧交叉比對、
   去蕪存菁，最終產出一份結構精美的統整報告。

架構示意：
┌──────────────┐   ┌──────────────┐
│  🏛️ NQU 官網  │   │  💬 Dcard 版  │
│  (BeautifulSoup) │   │ (Perplexity)  │
└──────┬───────┘   └──────┬───────┘
       │    並行抓取     │
       └──────┬──────────┘
              ▼
     ┌────────────────┐
     │  🤖 Gemini Pro  │
     │   智慧總結引擎   │
     └────────┬───────┘
              ▼
        📋 統整報告
"""

import os
import re
import requests
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("dcard_search_tool")

# ── API Keys ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── NQU 官網 URL 常數 ──
NQU_BASE_URL = "https://www.nqu.edu.tw"
NQU_ANNOUNCEMENTS_URL = f"{NQU_BASE_URL}/p/422-1000-2.php?Lang=zh-tw"
NQU_ACADEMIC_NEWS_URL = f"{NQU_BASE_URL}/p/402-1000-5-1.php?Lang=zh-tw"


# =============================================================================
# 🛡️ Discord Embed 抑制器
# =============================================================================

def _suppress_discord_embeds(text: str) -> str:
    """
    將文字中所有裸露的 URL 包在 <角括號> 中，抑制 Discord 自動生成的連結預覽方塊。
    同時處理 Markdown 連結格式：[text](url) → [text](<url>)
    """
    # 步驟 1：處理 Markdown 連結 [text](url) → [text](<url>)
    # 避免重複包裝已經有 <> 的
    text = re.sub(
        r'\[([^\]]+)\]\(((?!<)https?://[^)]+)\)',
        r'[\1](<\2>)',
        text
    )
    
    # 步驟 2：處理裸露 URL（不在 Markdown 連結內、不已被 <> 包裝的）
    text = re.sub(
        r'(?<!\()(?<!<)(https?://[^\s>)\]]+)(?!>)(?!\))',
        r'<\1>',
        text
    )
    
    return text


# =============================================================================
# 📰 來源 1：NQU 官方公告爬蟲
# =============================================================================

def _scrape_nqu_page(url: str, max_items: int = 15) -> list[dict]:
    """
    爬取 NQU 官網公告列表頁面，回傳結構化的公告清單。
    支援兩種頁面結構：表格式（校內公告）與連結列表式（學術焦點）。
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("❌ 未安裝 beautifulsoup4，無法爬取官網公告")
        return []

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = "utf-8"
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"❌ 無法連線至 NQU 官網：{e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    # 策略 1：表格式公告（校內公告頁面）
    table = soup.find("table", class_="table")
    if table:
        rows = table.find_all("tr")
        for row in rows[:max_items]:
            cells = row.find_all("td")
            if len(cells) >= 2:
                link_tag = cells[1].find("a") if len(cells) > 1 else cells[0].find("a")
                if not link_tag:
                    link_tag = cells[0].find("a")
                if link_tag:
                    title = link_tag.get_text(strip=True)
                    href = link_tag.get("href", "")
                    if href and not href.startswith("http"):
                        href = NQU_BASE_URL + href
                    unit = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    date = cells[0].get_text(strip=True) if len(cells) > 2 else ""
                    if title:
                        results.append({"title": title, "url": href, "unit": unit, "date": date})
        return results

    # 策略 2：連結列表式（學術焦點動態等）
    content_area = soup.find("div", class_="mpgdetail") or soup.find("div", id="pageptlist")
    links = content_area.find_all("a") if content_area else soup.find_all("a")

    seen_titles = set()
    nav_blacklist = {"關於金大", "行政單位", "學術單位", "招生資訊", "教職員", "學生",
                     "新生專區", "一般訪客", "校長遴選", "網站導覽", "校訊", "English", "首頁", "跳到主要"}
    for a_tag in links:
        href = a_tag.get("href", "")
        title = a_tag.get_text(strip=True)
        if not title or not href or href.startswith("javascript:") or href.startswith("#"):
            continue
        if len(title) < 8 or any(nav in title for nav in nav_blacklist):
            continue
        if title not in seen_titles:
            seen_titles.add(title)
            if not href.startswith("http"):
                href = NQU_BASE_URL + href
            results.append({"title": title, "url": href, "unit": "", "date": ""})
            if len(results) >= max_items:
                break
    return results


def _fetch_nqu_news(query: str) -> list[dict]:
    """內部函式：爬取官網並根據關鍵字篩選，回傳原始結構化清單。"""
    all_items: list[dict] = []
    all_items.extend(_scrape_nqu_page(NQU_ANNOUNCEMENTS_URL, max_items=20))
    all_items.extend(_scrape_nqu_page(NQU_ACADEMIC_NEWS_URL, max_items=10))

    if not all_items:
        return []

    keywords = [k for k in re.split(r"[\s,，、]+", query.lower()) if len(k) >= 1]
    matched = [
        item for item in all_items
        if any(kw in (item["title"].lower() + " " + item.get("unit", "").lower()) for kw in keywords)
    ]
    # 如果沒匹配到，退回最新公告
    return matched[:15] if matched else all_items[:10]


# =============================================================================
# 💬 來源 2：Dcard 金大版搜尋（透過 OpenRouter / Perplexity）
# =============================================================================

def _fetch_dcard_raw(query: str) -> str:
    """
    內部函式：透過 Perplexity 搜尋 Dcard 金門大學版，回傳原始文字結果。
    Prompt 架構保持原有邏輯，僅美化排版。
    """
    if not OPENROUTER_API_KEY:
        return ""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "NQU-Course-Bot-Dcard-Search"
    }

    prompt = f"""
請搜尋 site:dcard.tw/f/nqu
找出金門大學版中與「{query}」相關的文章與留言，特別是關於推薦或評價的部分。

整理：
1. 被討論/推薦的教授名字
2. 推薦/評價原因
3. 是否有負評或需注意的地方
4. 文章連結

過濾規則：
1. 沒有建設性的留言不要列入，例如：
   - 「推某某老師」
   - 「+1」
   - 「好過」

2. 客觀統整出被推薦的教授與具體理由。

輸出表格：
教授 | 評價/推薦原因 | 負評/注意事項 | 來源文章連結

來源文章連結請輸出為 Markdown 超連結格式，例如：
[文章標題](https://www.dcard.tw/f/nqu/p/123456)

**重要規則**：
- 不要使用 [1][2] 這種引用編號。
- 每個來源都要附上完整 URL 且為有效的超連結。
- 如果找不到與「{query}」相關的具體評價，請直接回答「目前在 Dcard 金門大學版上找不到與『{query}』相關的詳細評價或推薦。」，不要硬掰。
"""

    data = {
        "model": "perplexity/sonar-pro",
        "messages": [
            {"role": "system", "content": "回答請使用繁體中文。你是一個專業的資料整理助手，專門整理 Dcard 上的課程與教授評價。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
        logger.warning(f"⚠️ Dcard API 回應異常 (status={response.status_code})")
        return ""
    except requests.exceptions.Timeout:
        logger.warning("⚠️ Dcard 搜尋超時")
        return ""
    except Exception as e:
        logger.warning(f"⚠️ Dcard 搜尋錯誤: {e}")
        return ""


# =============================================================================
# 🤖 總結引擎：Gemini 3.1 Pro 智慧交叉比對
# =============================================================================

def _gemini_synthesize(query: str, nqu_items: list[dict], dcard_text: str) -> str:
    """
    將兩個來源的原始資料餵給 Gemini 3.1 Pro，
    由 AI 進行智慧統整、去蕪存菁，產出一份精美的統合報告。
    """
    import config

    # ── 組裝來源資料 ──
    nqu_section = ""
    if nqu_items:
        nqu_lines = []
        for item in nqu_items:
            line = f"- {item['title']}"
            if item.get("date"):
                line += f" ({item['date']})"
            if item.get("unit"):
                line += f" [{item['unit']}]"
            if item.get("url"):
                line += f"\n  連結：{item['url']}"
            nqu_lines.append(line)
        nqu_section = "\n".join(nqu_lines)
    else:
        nqu_section = "（無相關結果）"

    dcard_section = dcard_text if dcard_text else "（無相關結果或搜尋失敗）"

    # ── Gemini 總結 Prompt ──
    synthesis_prompt = f"""你是國立金門大學的校園資訊整理專家。你剛剛收到了兩個來源的搜尋結果，請為學生做出一份精美、有用的統整報告。

## 🔍 學生的提問
「{query}」

## 📰 來源 1：金門大學官網公告
{nqu_section}

## 💬 來源 2：Dcard 金門大學版（學生討論區）
{dcard_section}

## 📋 你的任務
請根據以上兩個來源的資料，產出一份統整報告。格式要求：

1. **開頭**：用一句話總結搜尋結果概況。
2. **📰 官方資訊**：如果官網有相關公告，列出最重要的 3~5 則，每則包含標題與連結（Markdown 格式）。
3. **💬 學生口碑**：如果 Dcard 有相關討論，整理出關鍵評價（教授推薦、課程心得等）。保留原始的 Dcard 文章連結。
4. **💡 總結建議**：根據兩邊的資料，給學生一個簡短實用的建議。

## ⚠️ 重要規則
- 如果某個來源「無相關結果」，不要硬掰，直接跳過該區塊。
- 來源連結必須保留完整 URL，不要捏造連結。
- 語氣友善、簡潔，像學長姐在幫忙一樣。
- 如果兩個來源都沒有有用的資訊，坦白告知並建議學生去哪裡可以找到答案（例如教務處、系辦等）。
- 回覆使用繁體中文。
- 所有連結必須使用 Markdown 格式：[標題](<URL>)，注意 URL 外層要有尖括號 <>。
- 絕對不要輸出裸露的 URL（沒有 Markdown 格式包裝的），否則會在 Discord 產生雜亂的預覽卡片。"""

    payload = {
        "contents": [{"parts": [{"text": synthesis_prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": config.GEMINI_PRO_MAX_TOKENS,
            "thinkingConfig": {
                "thinkingLevel": "medium"
            }
        }
    }

    try:
        response = requests.post(config.GEMINI_API_URL, json=payload, timeout=config.GEMINI_PRO_TIMEOUT)
        response.raise_for_status()
        text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if text:
            return _suppress_discord_embeds(text)
    except Exception as e:
        logger.error(f"❌ Gemini 總結引擎失敗，退回原始資料：{e}")

    # ── Fallback：Gemini 失敗時，直接回傳原始資料 ──
    fallback_parts = []
    if nqu_items:
        fallback_parts.append("📰 **金門大學官網公告**\n")
        for i, item in enumerate(nqu_items[:8], 1):
            fallback_parts.append(f"**{i}.** [{item['title']}]({item['url']})")
    if dcard_text:
        fallback_parts.append("\n💬 **Dcard 金門大學版**\n")
        fallback_parts.append(dcard_text)
    if not fallback_parts:
        return f"❌ 抱歉，在金門大學官網和 Dcard 上都找不到與「{query}」相關的資料。"
    return _suppress_discord_embeds("\n".join(fallback_parts))


# =============================================================================
# 🔗 公開 API：統一搜尋入口
# =============================================================================

def search_campus_info(query: str) -> str:
    """
    🔗 統一校園資訊搜尋入口。
    並行爬取 NQU 官網 + Dcard 金大版，再由 Gemini 3.1 Pro 智慧總結。

    Args:
        query: 使用者查詢關鍵字

    Returns:
        str: Gemini 總結的統整報告（Markdown）
    """
    logger.info(f"🔗 啟動校園資訊聚合搜尋：「{query}」")

    nqu_items: list[dict] = []
    dcard_text: str = ""

    # ── 並行抓取兩個來源（節省 5~10 秒等待時間）──
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_nqu = executor.submit(_fetch_nqu_news, query)
        future_dcard = executor.submit(_fetch_dcard_raw, query)

        try:
            nqu_items = future_nqu.result(timeout=20)
            logger.info(f"📰 NQU 官網：取得 {len(nqu_items)} 筆結果")
        except Exception as e:
            logger.warning(f"⚠️ NQU 官網爬取失敗：{e}")

        try:
            dcard_text = future_dcard.result(timeout=65)
            logger.info(f"💬 Dcard：取得 {len(dcard_text)} 字結果")
        except Exception as e:
            logger.warning(f"⚠️ Dcard 搜尋失敗：{e}")

    # ── Gemini Pro 總結 ──
    logger.info("🤖 啟動 Gemini 3.1 Pro 總結引擎...")
    result = _gemini_synthesize(query, nqu_items, dcard_text)
    logger.info(f"✅ 校園資訊搜尋完成，回傳 {len(result)} 字")
    return result


# =============================================================================
# 🔁 向下相容：保留獨立呼叫的舊函式名稱
# =============================================================================

def search_nqu_news(query: str) -> str:
    """獨立搜尋 NQU 官網公告（向下相容 /nqu_news 指令）。"""
    items = _fetch_nqu_news(query)
    if not items:
        return "❌ 無法連線至金門大學官網或目前沒有任何公告，請稍後再試。"

    lines = [f"🏛️ 在**金門大學官網**上找到 **{len(items)}** 則相關公告：\n"]
    for i, item in enumerate(items, 1):
        line = f"**{i}.** [{item['title']}]({item['url']})"
        meta = []
        if item.get("date"):
            meta.append(f"📅 {item['date']}")
        if item.get("unit"):
            meta.append(f"🏢 {item['unit']}")
        if meta:
            line += f"\n　　{'　'.join(meta)}"
        lines.append(line)
    lines.append(f"\n🔗 [前往官網查看更多公告...](<{NQU_ANNOUNCEMENTS_URL}>)")
    return _suppress_discord_embeds("\n".join(lines))


def search_dcard_professor(query: str) -> str:
    """獨立搜尋 Dcard 教授評價（向下相容 /dcard_search 指令）。"""
    result = _fetch_dcard_raw(query)
    if not result:
        if not OPENROUTER_API_KEY:
            return "❌ 系統錯誤：未設定 OPENROUTER_API_KEY，請聯繫管理員處理。"
        return "❌ 搜尋 Dcard 時發生錯誤，請稍後再試。"
    return _suppress_discord_embeds(result)


# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    import config
    config.setup_logging()

    print("=" * 60)
    print("🔗 測試 統一校園搜尋（NQU + Dcard + Gemini 總結）")
    print("=" * 60)
    print(search_campus_info("招生"))
