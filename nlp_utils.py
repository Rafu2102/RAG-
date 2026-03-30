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
}


def get_ws_model():
    """
    取得 CKIP 斷詞模型（Singleton）。
    首次呼叫時自動下載模型權重（約 2GB）並載入。

    Returns:
        ckiptagger.WS 模型實例
    """
    global _ws_model
    if _ws_model is None:
        from ckiptagger import data_utils, WS

        # 自動下載模型權重（如果尚未下載）
        model_path = os.path.join(CKIP_DATA_DIR, "data")
        if not os.path.exists(model_path):
            logger.info("⬇️ 正在下載 CKIP Tagger 模型權重 (約 2GB，請耐心等候)...")
            data_utils.download_data_gdown(CKIP_DATA_DIR)

        logger.info("🧠 載入 CKIP 斷詞模型 (強制使用 CPU 以保護 8GB VRAM)...")
        # 【VRAM 防爆】強制 disable_cuda=True，把珍貴的 VRAM 留給 LLM 和 Reranker
        _ws_model = WS(model_path, disable_cuda=True)
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
    必須把所有字串打包成 list 一次送進去，速度快 10 倍以上。

    Args:
        texts: 待斷詞的文本列表
        custom_dict: CKIP 強制斷詞字典（可選）

    Returns:
        list[list[str]]，每筆文本的斷詞結果
    """
    if not texts:
        return []

    ws = get_ws_model()

    # 執行斷詞（可選傳入強制字典）
    if custom_dict is not None:
        word_sentence_list = ws(texts, coerce_dictionary=custom_dict)
    else:
        word_sentence_list = ws(texts)

    # 清理空白字元
    cleaned_corpus = []
    for words in word_sentence_list:
        cleaned_corpus.append([w for w in words if w.strip()])

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
