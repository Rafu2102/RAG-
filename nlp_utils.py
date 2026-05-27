 # -*- coding: utf-8 -*-
"""
nlp_utils.py — CKIP Tagger 分詞與 NLP 處理中樞
================================================
使用中研院 CKIP Tagger（State-of-the-Art 繁體中文 NLP 模型）。
核心策略：
  1. Singleton 模式 — 全專案共用一份 WS 模型，避免重複載入 2GB 權重
  2. 強制 CPU 模式 — disable_cuda=True，VRAM 留給 LLM + Reranker
  3. 批次處理 — 建索引時一次送入所有文本，速度快 10x
  4. 強制字典 — coerce_dictionary 絕對控制，教師/課程名不被切碎
"""
import os
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

# CKIP 模型存放路徑（放在 data/ 目錄下，與課程資料並列）
CKIP_DATA_DIR = os.path.join(config.PROJECT_ROOT, "data")

# Singleton 模型實例
_ws_model = None

# 繁體中文停用詞表（BM25 及搜尋用）
STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一個", "上", "也", "很", "到", "說", "要", "去",
    "你", "會", "著", "沒有", "看", "好", "自己", "這", "他", "她",
    "們", "那", "個", "與", "及", "或", "但", "而", "將", "已",
    "為", "從", "以", "可", "能", "之", "於", "等", "被", "其",
    "把", "所", "對", "讓", "這個", "那個", "什麼", "怎麼", "哪",
    "\r", "\n", "\t", " ", "：", "，", "。", "、", "（", "）",
    "請問", "查詢", "想知道", "課表", "教授", "課程", "老師", "有沒有",
    "開課", "在哪裡", "什麼時候", "怎麼去", "哪個", "誰有", "有人",
    "知道", "你好", "哈囉", "嗨", "請問一下", "想問", "幫我", "找",
    "哪裡", "何時", "是誰", "誰", "怎麼", "如何", "哪一個", "哪一門",
    "我想", "我想要", "能不能", "可以嗎", "可以", "請", "謝謝", "感謝",
    "一下", "一些", "一些些", "還有", "或者是", "或是", "以及", "以及說",
    "等等", "之類的", "的話", "有些", "因此", "所以", "如果", "假如",
    "那麼", "這時候", "那時候", "這款", "那款", "這些", "那些", "這兒",
    "那兒", "這樣子", "這樣", "那樣", "那樣子", "有些時候", "有時候",
    "通常", "一般而言", "總之", "大致上", "主要", "各位", "大家", "喔",
    "啦", "吧", "呢", "啊", "咩", "耶", "囉", "唷", "哈", "喔的"
}


def get_ws_model():
    """
    取得 CKIP 斷詞模型（Singleton）。
    
    基於高併發與生產環境安全策略，此處已移除執行時期（Runtime）的同步下載邏輯。
    如果本地端不存在模型權重，將拋出 FileNotFoundError 並附帶明確的使用者部署與下載指引。

    Returns:
        ckiptagger.WS 模型實例
    """
    global _ws_model
    if _ws_model is None:
        from ckiptagger import WS

        model_path = os.path.join(CKIP_DATA_DIR, "data")
        if not os.path.exists(model_path):
            error_msg = (
                f"❌ [CKIP Tagger 啟動失敗]\n"
                f"找不到中研院 CKIP Tagger 模型權重目錄：{model_path}\n\n"
                f"💡 【部署與下載指引】：\n"
                f"1. 請勿在執行時期（Runtime）動態下載 2 GB 的模型，以免造成系統停頓或連線超時。\n"
                f"2. 請在部署或啟動服務前，於主機上執行以下手動下載腳本：\n"
                f"   python -c \"from ckiptagger import data_utils; data_utils.download_data_gdown('{CKIP_DATA_DIR}')\"\n"
                f"3. 確保下載完成後，該目錄下有 '{model_path}' 子目錄且內容完整（無損壞的 ZIP 殘留）。\n"
            )
            logger.critical(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("🧠 載入 CKIP 斷詞模型 (嘗試使用 GPU)...")
        # 根據使用者指示解除 VRAM 保護，允許 CKIP 進入 GPU 運算
        _ws_model = WS(model_path, disable_cuda=False)
        logger.info("✅ CKIP 斷詞模型載入完成")

    return _ws_model


def build_ckip_dictionary(words_list: list[str]):
    """
    建立 CKIP 強制斷詞字典（取代 jieba.add_word）。
    將金大的課程名稱與老師名稱設定為最高權重，
    CKIP 的 coerce_dictionary 是絕對權威，說不切就不切！

    Args:
        words_list: 需要保護不被切碎的詞彙列表

    Returns:
        ckiptagger 格式的強制字典物件
    """
    from ckiptagger import construct_dictionary
    word_to_weight = {word: 1 for word in words_list if word.strip()}
    if word_to_weight:
        logger.info(f"📝 CKIP 強制字典建構完成：{len(word_to_weight)} 個實體詞彙")
    return construct_dictionary(word_to_weight)


def tokenize_texts_ckip(texts: list[str], custom_dict=None) -> list[list[str]]:
    """
    【批次處理】使用 CKIP 對多筆文本進行斷詞。
    深度學習模型啟動成本高，切忌用 for 迴圈一筆一筆送！
    但若一次送入數萬筆（如 30,000+），會因為 LSTM padding 機制導致記憶體暴增與效能極低。
    因此採用適當的批次大小（如 500 筆）進行分批處理，平衡啟動開銷與 padding 開銷。

    Args:
        texts: 待斷詞的文本列表
        custom_dict: CKIP 強制斷詞字典（可選）

    Returns:
        list[list[str]]，每筆文本的斷詞結果
    """
    if not texts:
        return []

    ws = get_ws_model()

    batch_size = 500
    cleaned_corpus = []
    total = len(texts)
    
    logger.info(f"🧠 開始 CKIP 批次分詞，總文本數：{total}，批次大小：{batch_size}")
    
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        
        # 執行斷詞（可選傳入強制字典）
        if custom_dict is not None:
            word_sentence_list = ws(batch, coerce_dictionary=custom_dict)
        else:
            word_sentence_list = ws(batch)

        # 清理空白字元
        for words in word_sentence_list:
            cleaned_corpus.append([w for w in words if w.strip()])
            
        completed = min(i + batch_size, total)
        logger.info(f"  ⚡ CKIP 分詞進度：{completed}/{total} ({(completed/total)*100:.1f}%)")

    return cleaned_corpus


def tokenize_chinese(text: str) -> list[str]:
    """
    使用 CKIP 進行中文分詞，並過濾停用詞和短詞。
    
    此函式提供與舊版 jieba tokenize_chinese 相容的介面，
    方便 retriever.py 等模組無痛切換。

    Args:
        text: 待斷詞的中文文本

    Returns:
        過濾後的詞彙列表
    """
    results = tokenize_texts_ckip([text])
    if not results:
        return []
    
    words = results[0]
    return [w.strip() for w in words if w.strip() and len(w.strip()) > 1 and w not in STOPWORDS]
