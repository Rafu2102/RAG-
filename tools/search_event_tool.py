import json
from pathlib import Path
import jieba

BASE_DIR = Path(__file__).resolve().parents[1]
EVENTS_PATH = BASE_DIR / "data" / "events.json"

with EVENTS_PATH.open(encoding="utf-8") as f:
    EVENTS = json.load(f)

def search_academic_events(query: str) -> list[dict]:
    """使用 jieba 萃取關鍵字來搜尋學校行事曆事件"""
    
    # 【同義詞對照表】解決學生俗稱與官方行事曆名稱不匹配的問題
    SYNONYM_MAP = {
        "退選": "停修",
        "加退選": "選課 停修",
        "加選": "選課",
        "選課": "選課",
        "宿舍": "宿舍",
        "住宿": "宿舍",
        "開學": "上課開始",
        "註冊": "繳費",
        "學費": "繳費",
        "期中": "期中考試",
        "期末": "期末考試"
    }
    
    # 進行同義詞替換擴充
    expanded_query = query
    for slang, official in SYNONYM_MAP.items():
        if slang in query:
            expanded_query += f" {official}"
            
    keywords = list(jieba.cut(expanded_query))
    stopwords = {"什麼時候", "幫我", "加到", "行事曆", "的", "請問", "日期", "時間", "是", "何時", "查詢", "我想"}
    valid_keywords = [kw for kw in keywords if kw not in stopwords and len(kw) >= 2]
    
    if not valid_keywords:
        # 如果無法切出有效關鍵字，嘗試直接 substring 搜尋
        results = [e for e in EVENTS if query in e.get("title", "")]
        return results[:3]

    scored_events = []
    for e in EVENTS:
        title = e.get("title", "")
        # 計算匹配分數：只要有效關鍵字出現在官方行事曆標題中即加分
        score = sum(1 for kw in valid_keywords if kw in title)
        if score > 0:
            scored_events.append((score, e))
            
    scored_events.sort(key=lambda x: x[0], reverse=True)
    return [e[1] for e in scored_events[:3]]