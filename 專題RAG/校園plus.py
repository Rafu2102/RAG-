# -*- coding: utf-8 -*-
import os
import re
import json
import uuid
import shutil
import requests
import chromadb
from typing import List, Dict, Any, Tuple

# =============== 使用者環境設定 ===============
JSON_PATH = r"D:\網頁\RAG\knowledge\資工系113學年度第2學期課程資訊.json"

CHROMA_PATH = "db/chroma_demo"
COLLECTION_NAME = "collection_v2"

# Ollama
OLLAMA_HOST = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
GEN_MODEL_CANDIDATES = [
    "llama3.1:8b",
    "llama3.1:8b-instruct",
    "qwen2.5:7b-instruct",
    "gemma2:9b-instruct",
    "deepseek-r1:7b"
]
TEMPERATURE = 0.1
REQUEST_TIMEOUT = 120

# 檢索（僅在你開啟後備 RAG 時會用）
TOPN_VECTOR = 20
TOPK_AFTER_RERANK = 6

# 互動旗標
SHOW_CONTEXT = False             # /ctx on|off
ALLOW_RAG_FALLBACK = False       # 預設禁用後備 RAG（避免亂補）
STRICT_UNIQUE = True             # 多筆很像時要你縮小，以免配錯
MAX_CONTEXT_PREVIEW_CHARS = 700

# 全域資料
COURSE_TITLES: List[str] = []
DATASET: List[Dict[str, Any]] = []

# =============== 小工具 ===============
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def http_post_json(url: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def pick_first_available_model(candidates: List[str]) -> str:
    test_prompt = "ok"
    for m in candidates:
        try:
            _ = http_post_json(
                f"{OLLAMA_HOST}/api/generate",
                {"model": m, "prompt": test_prompt, "stream": False, "options": {"temperature": 0.0}},
                timeout=15
            )
            return m
        except Exception:
            continue
    return candidates[-1]

def is_str(x) -> bool:
    return isinstance(x, str)

def force_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return [s for s in str(v).splitlines() if s.strip()]

def clean_str(s: Any) -> str:
    s = "" if s is None else str(s)
    return s.strip()

def g(course: Dict[str, Any], key: str, default: Any = "未標註") -> Any:
    v = course.get(key, default)
    if v in (None, ""):
        return default
    return v

# 嚴格正規化：只保留中文、英數，其他全移除（含標點/空白）
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return s

def char_ngrams(s: str, n: int = 2) -> List[str]:
    s = normalize_text(s)
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

def dice_coeff(a: str, b: str, n: int = 2) -> float:
    A = set(char_ngrams(a, n))
    B = set(char_ngrams(b, n))
    if not A and not B:
        return 1.0
    return 2 * len(A & B) / (len(A) + len(B))

# =============== JSON 清洗（修復小缺陷） ===============
def clean_course_record(course: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(course)
    # 字串欄位去頭尾空白
    for k in ["學年度","學期","課程名稱","部別","開課班級","學分數","授課時數","授課教師","必選修",
              "永續發展目標(SDGs)","永續發展目標 (SDGs)","大學社會責任(USR)關聯性","大學社會責任 (USR) 關聯性",
              "教學目標"]:
        if k in out and is_str(out[k]):
            out[k] = clean_str(out[k])
    # 列表欄位統一成 list[str]
    for k in ["教學綱要","教科書","參考書","教學進度表","成績評定方式","課堂要求"]:
        if k in out:
            out[k] = force_list(out[k])
    return out

# =============== 教科書群組化解析 ===============
BOOK_HEAD = re.compile(r"^\s*(\d+)[\.\、]\s*")  # "1." "2、"
FIELD_KV = re.compile(r"^\s*([A-Za-z\u4e00-\u9fa5()（）]+)：\s*(.*)$")

def parse_textbooks(raw_list: List[str]) -> List[Dict[str, str]]:
    if not raw_list:
        return []
    groups: List[List[str]] = []
    cur: List[str] = []
    seen_group = False

    for line in raw_list:
        s = clean_str(line)
        if not s:
            continue
        if BOOK_HEAD.match(s):
            seen_group = True
            if cur:
                groups.append(cur)
            cur = [s]
        else:
            if cur:
                cur.append(s)
            else:
                cur = [s]
    if cur:
        groups.append(cur)

    items: List[Dict[str, str]] = []
    if seen_group:
        for grp in groups:
            item: Dict[str, str] = {}
            head = grp[0]
            m = BOOK_HEAD.match(head)
            if m:
                item["編號"] = m.group(1)
                head = BOOK_HEAD.sub("", head).strip()
            m2 = FIELD_KV.match(head)
            if m2:
                item[m2.group(1)] = m2.group(2)
            elif head:
                item["描述"] = head
            for rest in grp[1:]:
                m3 = FIELD_KV.match(rest)
                if m3:
                    item[m3.group(1)] = m3.group(2)
                else:
                    item.setdefault("描述補充", "")
                    item["描述補充"] += (("\n" if item["描述補充"] else "") + rest)
            items.append(item)
    else:
        for s in raw_list:
            s = clean_str(s)
            if s:
                items.append({"引用": s})
    return items

def format_textbooks(items: List[Dict[str, str]]) -> str:
    if not items:
        return "未標註"
    out = []
    for i, it in enumerate(items, 1):
        parts = []
        for k in ["書名","作者","出版社","出版日期","版本","引用","描述"]:
            if it.get(k):
                parts.append(f"{k}：{it[k]}")
        for k, v in it.items():
            if k not in ["編號","書名","作者","出版社","出版日期","版本","引用","描述","描述補充"]:
                parts.append(f"{k}：{v}")
        if it.get("描述補充"):
            parts.append(it["描述補充"])
        out.append(f"{i}. " + "；".join(parts) if parts else f"{i}. 未標註")
    return "\n".join(out)

# =============== JSON → Chunks（for RAG 備用） ===============
def build_base_block(course: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"[索引] 班級={g(course,'開課班級')} 課名={str(g(course,'課程名稱')).strip()} 老師={g(course,'授課教師')} 學年度={g(course,'學年度')} 學期={g(course,'學期')} 必選修={g(course,'必選修')}")
    lines.append("【課程基本資料】")
    lines.append(f"學年度：{g(course,'學年度')}")
    lines.append(f"學期：{g(course,'學期')}")
    lines.append(f"課程名稱：{str(g(course,'課程名稱')).strip()}")
    lines.append(f"部別：{g(course,'部別')}")
    lines.append(f"開課班級：{g(course,'開課班級')}")
    lines.append(f"學分數：{g(course,'學分數')}")
    lines.append(f"授課時數：{g(course,'授課時數')}")
    lines.append(f"授課教師：{g(course,'授課教師')}")
    lines.append(f"必選修：{g(course,'必選修')}")
    lines.append("【教學目標】")
    lines.append(clean_str(g(course, '教學目標', '未標註')))
    lines.append("【教學綱要】")
    lines.append("\n".join(force_list(course.get('教學綱要', []))))
    lines.append("【教科書】")
    lines.append("\n".join(force_list(course.get('教科書', []))))
    lines.append("【參考書】")
    lines.append("\n".join(force_list(course.get('參考書', []))))
    return "\n".join([x for x in lines if str(x).strip()])

def split_progress_blocks(course: Dict[str, Any]) -> List[str]:
    progress: List[str] = force_list(course.get("教學進度表", []))
    score_start = None
    for i, line in enumerate(progress):
        if "成績評定方式" in str(line):
            score_start = i
            break
    weeks = progress if score_start is None else progress[:score_start]
    scores = [] if score_start is None else progress[score_start:]

    def block(label: str, week_lines: List[str]) -> str:
        header = f"[索引] 進度段={label} 班級={g(course,'開課班級')} 課名={str(g(course,'課程名稱')).strip()} 老師={g(course,'授課教師')}"
        body = "\n".join(week_lines)
        tail = ""
        if label == "週10-18" and scores:
            tail = "\n" + "\n".join(scores)
        return "\n".join([header, "【教學進度表】", body, tail]).strip()

    first_half, second_half = [], []
    for raw in weeks:
        s = clean_str(raw)
        m = re.match(r"^\s*(\d+)", s)
        num = int(m.group(1)) if m else None
        if num is not None and num >= 10:
            second_half.append(s)
        else:
            first_half.append(s)

    if not first_half and not second_half:
        mid = len(weeks) // 2
        first_half, second_half = weeks[:mid], weeks[mid:]

    out = []
    if first_half: out.append(block("週1-9", first_half))
    if second_half: out.append(block("週10-18", second_half))
    return out

def load_json_chunks(json_path: str) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)

    data = [clean_course_record(c) for c in raw]

    chunks, metas = [], []
    for course in data:
        base = build_base_block(course)
        chunks.append(base)
        metas.append({"課程名稱": str(g(course, "課程名稱")).strip()})
        for pb in split_progress_blocks(course):
            chunks.append(pb)
            metas.append({"課程名稱": str(g(course, "課程名稱")).strip(), "段落": "教學進度"})
    return chunks, metas, data

# =============== 初始化 ===============
def initial():
    global COURSE_TITLES, DATASET
    print("初始化資料庫：重新建立 Chroma 集合並嵌入 JSON 課程資料...")
    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    ensure_dir(CHROMA_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents, metadatas, dataset = load_json_chunks(JSON_PATH)
    DATASET = dataset
    COURSE_TITLES = [str(g(c, "課程名稱")).strip() for c in DATASET]
    print(f"段落數：{len(documents)}；課程數：{len(COURSE_TITLES)}")

    ids, embeddings, kept_docs, kept_metas = [], [], [], []
    for i, doc in enumerate(documents):
        try:
            emb = ollama_embed(doc, purpose="doc")
        except Exception as e:
            print(f"[警告] 第 {i} 段嵌入失敗：{e}")
            continue
        kept_docs.append(doc)
        kept_metas.append(metadatas[i])
        embeddings.append(emb)
        ids.append(str(uuid.uuid4()))
    collection.add(ids=ids, documents=kept_docs, embeddings=embeddings, metadatas=kept_metas)
    print(f"完成：成功寫入 {len(kept_docs)} 段。")

# =============== 從問句抽課名片語 ===============
NOISE_PHRASES = {
    # 常見功能詞／問句詞
    "請問", "問題", "是誰", "是什麼", "在哪", "在哪裡", "安排", "進度", "考試",
    "第幾週", "幾週", "第", "週", "嗎", "呢",
    # 屬性詞（非課名）：之前漏了 USR/SDGs 相關
    "老師", "授課教師", "學分", "教科書", "教材",
    "成績評定方式", "評分", "評量", "比例",
    "教學進度表", "週次", "期中", "期末",
    "大學社會責任", "關聯性", "USR", "usr",
    "永續發展目標", "SDGs", "sdgs",
    "課程"
}

# 片語尾端常見助詞
TAIL_PARTICLES = ("的", "之")

def _strip_particles(s: str) -> str:
    s = s.strip()
    # 連續去掉尾端助詞（例：XXX的、XXX之）
    while any(s.endswith(p) for p in TAIL_PARTICLES):
        s = s[:-1].strip()
    return s

def extract_title_phrases_from_question(q: str) -> List[str]:
    # 連續中文（>=2）與英文詞（>=3，允許空白/連字號）
    ch = re.findall(r"[\u4e00-\u9fff]{2,}", q)
    en = re.findall(r"[A-Za-z][A-Za-z \-]{2,}", q)
    raw_parts = ch + en

    phrases: List[str] = []
    for s in raw_parts:
        s2 = s.strip()
        if not s2:
            continue

        # 若包含任一雜訊詞：嘗試把雜訊拿掉後再保留剩餘主體
        if any(n in s2 for n in NOISE_PHRASES):
            t = s2
            for n in NOISE_PHRASES:
                t = t.replace(n, "")
            t = _strip_particles(t)
            t = t.strip("：: ，, 的之")  # 再掃一次標點與助詞
            if len(t) >= 2:
                phrases.append(t)
            continue

        # 一般情況也去尾助詞
        s2 = _strip_particles(s2)
        # 像「離散數學的老師」會被上面處理，但保險起見再 split 一次
        if "的" in s2:
            left = _strip_particles(s2.split("的", 1)[0])
            right = _strip_particles(s2.split("的", 1)[-1])
            if len(left) >= 2 and left not in NOISE_PHRASES:
                phrases.append(left)
            # 右半通常是屬性（老師/學分⋯）會被 NOISE 擋掉；保險判斷
            if len(right) >= 2 and right not in NOISE_PHRASES:
                phrases.append(right)
        else:
            if s2 not in NOISE_PHRASES and len(s2) >= 2:
                phrases.append(s2)

    # 去重＆保持順序
    seen, out = set(), []
    for p in phrases:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out

# =============== 唯一對齊課名（子字串優先 + Dice 備援） ===============
def resolve_unique_course(q: str) -> Tuple[Dict[str, Any], List[Tuple[str, float, int]]]:
    phrases = extract_title_phrases_from_question(q)
    if not phrases:
        phrases = [q]

    # 先做「子字串」命中（強信號）：命中一個就視為唯一
    contains_scores = []
    for idx, title in enumerate(COURSE_TITLES):
        tnorm = normalize_text(title)
        score = 0.0
        for p in phrases:
            pnorm = normalize_text(p)
            if not pnorm:
                continue
            if pnorm in tnorm or tnorm in pnorm:
                # 片語越長權重越大
                score += max(1.0, len(pnorm) / 4.0)
        if score > 0.0:
            contains_scores.append((title, score, idx))

    if contains_scores:
        contains_scores.sort(key=lambda x: x[1], reverse=True)
        # 只有一個直接唯一；多個但第一名優勢 >= 1.0 也判唯一
        if len(contains_scores) == 1 or (contains_scores[0][1] - contains_scores[1][1] >= 1.0) or not STRICT_UNIQUE:
            return DATASET[contains_scores[0][2]], contains_scores[:5]
        else:
            return None, contains_scores[:5]

    # 沒子字串 → Dice（取各片語最大值）
    scores = []
    for idx, title in enumerate(COURSE_TITLES):
        s_max = 0.0
        for p in phrases:
            s_max = max(s_max, dice_coeff(p, title, n=2))
        scores.append((title, s_max, idx))
    scores.sort(key=lambda x: x[1], reverse=True)

    # 放寬一點門檻（避免正確課名被雜訊稀釋）
    if scores and ((scores[0][1] >= 0.58 and (len(scores) == 1 or scores[0][1] - scores[1][1] >= 0.08)) or not STRICT_UNIQUE):
        return DATASET[scores[0][2]], scores[:5]
    else:
        return None, scores[:5]


# =============== 進度解析（含期中/期末週次） ===============
def parse_progress(course: Dict[str, Any]) -> Dict[str, Any]:
    raw: List[str] = force_list(course.get("教學進度表", []))
    weeks, grading, reqs = [], [], []
    sdg = g(course, "永續發展目標(SDGs)", None) or g(course, "永續發展目標 (SDGs)", None)
    usr = g(course, "大學社會責任(USR)關聯性", None) or g(course, "大學社會責任 (USR) 關聯性", None)
    state = "weeks"
    for line in raw:
        s = clean_str(line)
        if not s: continue
        if s.startswith("成績評定方式"):
            state = "grading"; continue
        if s.startswith("課堂要求"):
            state = "reqs"; continue
        if ("永續發展目標" in s or "SDGs" in s) and not sdg:
            sdg = s.split("：",1)[-1].strip() if "：" in s else s; continue
        if "大學社會責任" in s and not usr:
            usr = s.split("：",1)[-1].strip() if "：" in s else s; continue
        if state == "weeks": weeks.append(s)
        elif state == "grading": grading.append(s)
        elif state == "reqs": reqs.append(s)

    mid_weeks, fin_weeks = [], []
    for w in weeks:
        if any(k in w for k in ["期中", "Midterm", "midterm"]):
            m = re.match(r"^\s*(\d+)", w)
            if m: mid_weeks.append(int(m.group(1)))
        if any(k in w for k in ["期末", "Final", "final"]):
            m = re.match(r"^\s*(\d+)", w)
            if m: fin_weeks.append(int(m.group(1)))

    return {
        "weeks": weeks,
        "grading": grading,
        "reqs": reqs,
        "sdg": sdg if sdg else "未標註",
        "usr": usr if usr else "未標註",
        "mid_weeks": sorted(set(mid_weeks)),
        "final_weeks": sorted(set(fin_weeks)),
    }

# =============== 抽取全部欄位（含多本教科書） ===============
def extract_all_fields(course: Dict[str, Any]) -> Dict[str, Any]:
    p = parse_progress(course)
    textbooks_items = parse_textbooks(force_list(course.get("教科書", [])))
    return {
        "學年度": clean_str(g(course, "學年度")),
        "學期": clean_str(g(course, "學期")),
        "課程名稱": clean_str(g(course, "課程名稱")),
        "部別": clean_str(g(course, "部別")),
        "開課班級": clean_str(g(course, "開課班級")),
        "學分數": clean_str(g(course, "學分數")),
        "授課時數": clean_str(g(course, "授課時數")),
        "授課教師": clean_str(g(course, "授課教師")),
        "必選修": clean_str(g(course, "必選修")),
        "教學目標": clean_str(g(course, "教學目標")),
        "教學綱要": force_list(course.get("教學綱要", [])),
        "教科書_items": textbooks_items,
        "參考書": force_list(course.get("參考書", [])),
        "教學進度表": p["weeks"],
        "成績評定方式": force_list(course.get("成績評定方式", [])) or p["grading"],
        "課堂要求": force_list(course.get("課堂要求", [])) or p["reqs"],
        "永續發展目標(SDGs)": p["sdg"],
        "大學社會責任(USR)關聯性": p["usr"],
        "mid_weeks": p["mid_weeks"],
        "final_weeks": p["final_weeks"],
    }

def format_textbooks_block(c: Dict[str, Any]) -> str:
    return format_textbooks(c.get("教科書_items", []))

# =============== 依問句重點回答（JSON 抽取） ===============
def list_to_lines(val: Any, prefix: str = "- ") -> str:
    if not val: return ""
    if isinstance(val, list):
        return "\n".join(f"{prefix}{str(x)}" for x in val if str(x).strip())
    return str(val)

def answer_from_course(q: str, c: Dict[str, Any]) -> str:
    want_textbook = ("教科書" in q) or ("教材" in q) or ("textbook" in q.lower())
    want_credits = ("學分" in q)
    want_teacher = any(k in q for k in ["老師","授課教師"])
    want_midweek = any(k in q for k in ["期中","midterm","Midterm"])
    want_finalweek = any(k in q for k in ["期末","final","Final"])
    want_progress = any(k in q for k in ["週次","進度","syllabus"])

    lines = [f"課程名稱：{c['課程名稱']}（{c['學年度']}／{c['學期']}）"]
    lines.append(f"開課班級：{c['開課班級']}　授課教師：{c['授課教師']}　學分數：{c['學分數']}　必選修：{c['必選修']}")

    if want_textbook:
        lines.append("教科書（全部列出）：")
        lines.append(format_textbooks_block(c))
    if want_credits:
        lines.append(f"學分數：{c['學分數']}")
    if want_teacher:
        lines.append(f"授課教師：{c['授課教師']}")
    if want_midweek:
        mw = c.get("mid_weeks", [])
        lines.append("期中考週次：" + (("、".join(str(x) for x in mw)) if mw else "未標註"))
    if want_finalweek:
        fw = c.get("final_weeks", [])
        lines.append("期末考週次：" + (("、".join(str(x) for x in fw)) if fw else "未標註"))
    if want_progress and not (want_midweek or want_finalweek):
        lines.append("教學進度表（週次）：")
        lines.append(list_to_lines(c.get("教學進度表", [])))

    if not any([want_textbook, want_credits, want_teacher, want_midweek, want_finalweek, want_progress]):
        lines.append("教學目標：")
        lines.append(c.get("教學目標","未標註"))
        lines.append("教學綱要：")
        lines.append(list_to_lines(c.get("教學綱要", [])))
        lines.append("教科書：")
        lines.append(format_textbooks_block(c))
        lines.append("參考書：")
        lines.append(list_to_lines(c.get("參考書", [])))
        lines.append("教學進度表：")
        lines.append(list_to_lines(c.get("教學進度表", [])))
        lines.append("成績評定方式：")
        lines.append(list_to_lines(c.get("成績評定方式", [])))
        lines.append("課堂要求：")
        lines.append(list_to_lines(c.get("課堂要求", [])))
        lines.append(f"SDGs：{c.get('永續發展目標(SDGs)','未標註')}")
        lines.append(f"USR 關聯性：{c.get('大學社會責任(USR)關聯性','未標註')}")

    lines.append("如需查詢其他課程，請提供關鍵字（課名/老師/班級）。")
    return "\n".join(lines).strip()

# =============== RAG 備用（預設關閉） ===============
def ollama_embed(text: str, purpose: str = "doc") -> List[float]:
    prefix = "passage: " if purpose == "doc" else "query: "
    payload = {"model": EMBED_MODEL, "prompt": prefix + text}
    res = http_post_json(f"{OLLAMA_HOST}/api/embeddings", payload, timeout=60)
    return res["embedding"]

def build_campus_guide_prompt(context: str, question: str) -> str:
    rules = "你是校園課程導覽員。只可使用〈校務資料〉；未命中即回「查無資料」。嚴禁臆測。以繁體中文作答。"
    return f"{rules}\n\n〈問題〉\n{question.strip()}\n\n〈校務資料〉\n{context.strip()}"

def rag_fallback_answer(qs: str) -> str:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(name=COLLECTION_NAME)
    qs_emb = ollama_embed(qs, purpose="query")
    res = col.query(query_embeddings=[qs_emb], query_texts=[qs], n_results=TOPN_VECTOR)
    docs = res.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(docs[:TOPK_AFTER_RERANK])
    prompt = build_campus_guide_prompt(context, qs)
    model = pick_first_available_model(GEN_MODEL_CANDIDATES)
    out = http_post_json(
        f"{OLLAMA_HOST}/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": TEMPERATURE}},
        timeout=REQUEST_TIMEOUT
    ).get("response","").strip()
    if SHOW_CONTEXT:
        preview = "\n\n".join(d[:MAX_CONTEXT_PREVIEW_CHARS] + ("…" if len(d) > MAX_CONTEXT_PREVIEW_CHARS else "") for d in docs[:TOPK_AFTER_RERANK])
        return f"[Model: {model}]\n{out}\n\n==== Context 預覽 ====\n{preview}"
    return f"[Model: {model}]\n{out}"

# =============== 單次查詢 ===============
def single_query(qs: str) -> str:
    # 先做課名唯一對齊（子字串優先 + Dice 備援）
    course, cand = resolve_unique_course(qs)
    if course is None:
        if STRICT_UNIQUE and cand:
            listing = "\n".join([f"{i+1}. {t}（相似度 {s:.2f}）" for i, (t, s, _) in enumerate(cand)])
            return f"[需要更明確的課名]\n找到多筆相近課程，為避免誤答請指定更精確的課名/班級/老師：\n{listing}"
        if ALLOW_RAG_FALLBACK:
            return rag_fallback_answer(qs)
        else:
            return "查無資料或課名不唯一，請補充（課名/班級/老師）。"

    fields = extract_all_fields(course)
    return "[Exact JSON]\n" + answer_from_course(qs, fields)

# =============== 互動式 CLI ===============
def chat_loop():
    global SHOW_CONTEXT, ALLOW_RAG_FALLBACK, STRICT_UNIQUE
    print("\n=== 校園課程導覽（Exact-First, Improved Title Matching） ===")
    print("指令：/help  /ctx on|off  /fallback on|off  /unique on|off  /reload  /exit\n")

    while True:
        try:
            qs = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見～")
            break

        if not qs:
            continue

        low = qs.lower()
        if low in ("/exit","exit","quit",":q"):
            print("再見～"); break
        if low == "/help":
            print("指令：/help  /ctx on|off  /fallback on|off  /unique on|off  /reload  /exit")
            print("說明：先抽課名片語→子字串優先匹配→唯一即直接用 JSON 抽取，不經 LLM。")
            continue
        if low.startswith("/ctx"):
            parts = qs.split()
            if len(parts)==2 and parts[1].lower() in ("on","off"):
                SHOW_CONTEXT = (parts[1].lower()=="on")
                print(f"[設定] Context 預覽 = {'開啟' if SHOW_CONTEXT else '關閉'}")
            else:
                print("用法：/ctx on|off")
            continue
        if low.startswith("/fallback"):
            parts = qs.split()
            if len(parts)==2 and parts[1].lower() in ("on","off"):
                ALLOW_RAG_FALLBACK = (parts[1].lower()=="on")
                print(f"[設定] 後備 RAG = {'開啟' if ALLOW_RAG_FALLBACK else '關閉'}")
            else:
                print("用法：/fallback on|off")
            continue
        if low.startswith("/unique"):
            parts = qs.split()
            if len(parts)==2 and parts[1].lower() in ("on","off"):
                STRICT_UNIQUE = (parts[1].lower()=="on")
                print(f"[設定] 嚴格唯一 = {'開啟' if STRICT_UNIQUE else '關閉'}")
            else:
                print("用法：/unique on|off")
            continue
        if low == "/reload":
            try:
                initial()
                print("[完成] 已重新載入 JSON 並重建向量庫。")
            except Exception as e:
                print(f"[錯誤] 重建失敗：{e}")
            continue

        try:
            ans = single_query(qs)
            print(ans)
        except Exception as e:
            print(f"[錯誤] 查詢/生成失敗：{e}")

# =============== 入口 ===============
if __name__ == "__main__":
    initial()
    chat_loop()
