# -*- coding: utf-8 -*-
import os
import re
import json
import uuid
import shutil
import requests
import chromadb
from typing import List, Dict, Any, Tuple

# ===================== 使用者環境設定 =====================
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
TOPN_VECTOR = 12
TOPK_AFTER_RERANK = 5
REQUEST_TIMEOUT = 120

# 行為開關（可用指令動態切）
EXACT_MODE = True            # 命中唯一課名 → 直接用 JSON 抽取
STRICT_UNIQUE = True         # 課名不唯一 → 不回答，列候選
DEBUG = False                # 除錯輸出

# ===================== 小工具 =====================
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def list_to_lines(val: Any, prefix: str = "- ") -> str:
    if not val:
        return ""
    if isinstance(val, list):
        return "\n".join(f"{prefix}{str(x)}" for x in val if str(x).strip())
    return str(val)

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

def g(course: Dict[str, Any], key: str, default: Any = "未標註") -> Any:
    v = course.get(key, default)
    if v in (None, ""):
        return default
    return v

def force_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return [s for s in str(v).splitlines() if s.strip()]

def clean_str(s: Any) -> str:
    s = "" if s is None else str(s)
    return s.strip()

# 嚴格正規化：只留中英數
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return s

# ===================== 教科書群組化（多本） =====================
BOOK_HEAD = re.compile(r"^\s*(\d+)[\.\、]\s*")
FIELD_KV = re.compile(r"^\s*([A-Za-z\u4e00-\u9fa5()（）]+)：\s*(.*)$")

def parse_textbooks(raw_list: List[str]) -> List[Dict[str, str]]:
    if not raw_list:
        return []
    groups, cur, seen_group = [], [], False
    for line in raw_list:
        s = clean_str(line)
        if not s: continue
        if BOOK_HEAD.match(s):
            seen_group = True
            if cur: groups.append(cur)
            cur = [s]
        else:
            if cur: cur.append(s)
            else: cur = [s]
    if cur: groups.append(cur)

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
            if it.get(k): parts.append(f"{k}：{it[k]}")
        for k, v in it.items():
            if k not in ["編號","書名","作者","出版社","出版日期","版本","引用","描述","描述補充"]:
                parts.append(f"{k}：{v}")
        if it.get("描述補充"): parts.append(it["描述補充"])
        out.append(f"{i}. " + "；".join(parts) if parts else f"{i}. 未標註")
    return "\n".join(out)

# ===================== 進度解析（期中/期末週次等） =====================
def parse_progress(course: Dict[str, Any]) -> Dict[str, Any]:
    raw: List[str] = force_list(course.get("教學進度表", []))
    weeks, grading, reqs = [], [], []
    sdg = g(course, "永續發展目標(SDGs)", None) or g(course, "永續發展目標 (SDGs)", None)
    usr = g(course, "大學社會責任(USR)關聯性", None) or g(course, "大學社會責任 (USR) 關聯性", None)
    state = "weeks"
    for line in raw:
        s = clean_str(line)
        if not s: continue
        if s.startswith("成績評定方式"): state = "grading"; continue
        if s.startswith("課堂要求"): state = "reqs"; continue
        if ("永續發展目標" in s or "SDGs" in s) and not sdg:
            sdg = s.split("：",1)[-1].strip() if "：" in s else s; continue
        if "大學社會責任" in s and not usr:
            usr = s.split("：",1)[-1].strip() if "：" in s else s; continue
        if state == "weeks": weeks.append(s)
        elif state == "grading": grading.append(s)
        elif state == "reqs": reqs.append(s)

    mid_weeks, fin_weeks = [], []
    for w in weeks:
        if any(k in w for k in ["期中","Midterm","midterm"]):
            m = re.match(r"^\s*(\d+)", w)
            if m: mid_weeks.append(int(m.group(1)))
        if any(k in w for k in ["期末","Final","final"]):
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

# ===================== JSON 清洗 & Chunks =====================
def clean_course_record(course: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(course)
    for k in ["學年度","學期","課程名稱","部別","開課班級","學分數","授課時數","授課教師","必選修",
              "永續發展目標(SDGs)","永續發展目標 (SDGs)",
              "大學社會責任(USR)關聯性","大學社會責任 (USR) 關聯性","教學目標"]:
        if k in out and isinstance(out[k], str):
            out[k] = clean_str(out[k])
    for k in ["教學綱要","教科書","參考書","教學進度表","成績評定方式","課堂要求"]:
        if k in out:
            out[k] = force_list(out[k])
    return out

def build_base_block(course: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"[索引] 系所={g(course,'系所','資工/電機')} 年級={g(course,'年級','未標註')} 班級={g(course,'開課班級')} 課名={g(course,'課程名稱')} 老師={g(course,'授課教師')} 學年度={g(course,'學年度')} 學期={g(course,'學期')} 必選修={g(course,'必選修')}")
    lines.append("【課程基本資料】")
    for k in ["學年度","學期","課程名稱","部別","開課班級","學分數","授課時數","授課教師","必選修"]:
        lines.append(f"{k}：{g(course,k)}")
    lines.append("【教學目標】"); lines.append(clean_str(g(course, "教學目標")))
    lines.append("【教學綱要】"); lines.append("\n".join(force_list(course.get("教學綱要", []))))
    lines.append("【教科書】"); lines.append("\n".join(force_list(course.get("教科書", []))))
    lines.append("【參考書】"); lines.append("\n".join(force_list(course.get("參考書", []))))
    return "\n".join([x for x in lines if str(x).strip()])

def split_progress_blocks(course: Dict[str, Any]) -> List[str]:
    progress: List[str] = force_list(course.get("教學進度表", []))
    score_start = None
    for i, line in enumerate(progress):
        if "成績評定方式" in str(line):
            score_start = i; break
    weeks = progress if score_start is None else progress[:score_start]
    scores = [] if score_start is None else progress[score_start:]

    def block(label: str, week_lines: List[str]) -> str:
        header = f"[索引] 進度段={label} 班級={g(course,'開課班級')} 課名={g(course,'課程名稱')} 老師={g(course,'授課教師')}"
        body = "\n".join(week_lines)
        tail = "\n" + "\n".join(scores) if (label=="週10-18" and scores) else ""
        return "\n".join([header, "【教學進度表】", body, tail]).strip()

    first_half, second_half = [], []
    for raw in weeks:
        s = clean_str(raw)
        m = re.match(r"^\s*(\d+)", s)
        num = int(m.group(1)) if m else None
        (second_half if (num and num>=10) else first_half).append(s)
    if not first_half and not second_half:
        mid = len(weeks)//2
        first_half, second_half = weeks[:mid], weeks[mid:]
    out = []
    if first_half: out.append(block("週1-9", first_half))
    if second_half: out.append(block("週10-18", second_half))
    return out

def load_json_chunks(json_path: str) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    with open(json_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    data = [clean_course_record(c) for c in raw]

    chunks, metas = [], []
    for course in data:
        base = build_base_block(course)
        chunks.append(base)
        metas.append({"課程名稱": g(course,"課程名稱")})
        for pb in split_progress_blocks(course):
            chunks.append(pb)
            metas.append({"課程名稱": g(course,"課程名稱"), "段落":"教學進度"})
    titles = [g(c,"課程名稱") for c in data]
    return chunks, metas, data, titles

# ===================== 嵌入與生成 =====================
def ollama_embed(text: str, purpose: str = "doc") -> List[float]:
    prefix = "passage: " if purpose == "doc" else "query: "
    payload = {"model": EMBED_MODEL, "prompt": prefix + text}
    res = http_post_json(f"{OLLAMA_HOST}/api/embeddings", payload, timeout=60)
    return res["embedding"]

def ollama_generate(prompt: str, model: str) -> str:
    res = http_post_json(
        f"{OLLAMA_HOST}/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": TEMPERATURE}},
        timeout=REQUEST_TIMEOUT
    )
    return res.get("response","").strip()

# ===================== 全域資料（Exact 用） =====================
DATASET: List[Dict[str, Any]] = []
COURSE_TITLES: List[str] = []

def initial():
    global DATASET, COURSE_TITLES
    print("初始化資料庫：重新建立 Chroma 集合並嵌入 JSON 課程資料...")
    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    ensure_dir(CHROMA_PATH)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents, metadatas, dataset, titles = load_json_chunks(JSON_PATH)
    DATASET = dataset
    COURSE_TITLES = titles
    print(f"段落數：{len(documents)}；課程數：{len(titles)}")

    ids, embeddings, kept_docs, kept_metas = [], [], [], []
    for i, doc in enumerate(documents):
        try:
            emb = ollama_embed(doc, purpose="doc")
        except Exception as e:
            print(f"[警告] 第 {i} 段嵌入失敗，跳過：{e}")
            continue
        kept_docs.append(doc)
        kept_metas.append(metadatas[i])
        embeddings.append(emb)
        ids.append(str(uuid.uuid4()))
    collection.add(ids=ids, documents=kept_docs, embeddings=embeddings, metadatas=kept_metas)
    print(f"完成：成功寫入 {len(kept_docs)} 段。")

# ===================== 課名片語抽取（強化） =====================
NOISE_PHRASES = {
    "請問","問題","是誰","是什麼","是甚麼","在哪","在哪裡","多少","安排","進度","考試",
    "第幾週","幾週","第","週","嗎","呢","吧","呀","請","想知道","告訴我","幫我",
    "老師","授課教師","學分","學分數","教科書","教材",
    "成績評定方式","評分","評量","比例",
    "教學進度表","週次","期中","期末",
    "大學社會責任","關聯性","usr","USR",
    "永續發展目標","sdgs","SDGs",
    "課程","用的是什麼","用的是甚麼","用什麼","用甚麼","然後","以及","與","跟","和"
}
TAIL_PARTICLES = ("的","之","用")
ATTR_TERMS_REGEX = r"(老師|授課教師|學分數?|教科書|教材|期中|期末|大學社會責任|USR|SDGs|關聯性|週次|進度)"

def _strip_particles(s: str) -> str:
    s = s.strip()
    while s and s.endswith(TAIL_PARTICLES):
        s = s[:-1].strip()
    return s

def _remove_noise(s: str) -> str:
    t = s
    for n in NOISE_PHRASES:
        t = t.replace(n, "")
    t = _strip_particles(t)
    t = t.strip("：:，,。.!？?、 ")
    return t

def extract_title_phrases_from_question(q: str) -> List[str]:
    phrases: List[str] = []
    if "《" in q and "》" in q:
        try:
            title = q.split("《",1)[1].split("》",1)[0].strip()
            if len(title) >= 2: phrases.append(title)
        except Exception: pass
    m = re.search(rf"([\u4e00-\u9fffA-Za-z0-9\-\s]{{2,}})的[^的]*{ATTR_TERMS_REGEX}", q)
    if m:
        cand = _remove_noise(m.group(1))
        if len(cand) >= 2: phrases.append(cand)
    ch_runs = re.findall(r"[\u4e00-\u9fff]{2,}", q)
    en_runs = re.findall(r"[A-Za-z][A-Za-z \-]{2,}", q)
    for s in ch_runs + en_runs:
        t = _remove_noise(s)
        if not t: continue
        if "的" in t:
            left = _remove_noise(t.split("的",1)[0])
            if len(left) >= 2: t = left
        is_en = bool(re.fullmatch(r"[A-Za-z0-9 \-]{4,}", t))
        if (not is_en and len(t) >= 3) or is_en:
            phrases.append(t)
    seen, out = set(), []
    for p in phrases:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out

def char_ngrams(s: str, n: int = 2) -> List[str]:
    s = normalize_text(s)
    if len(s) < n: return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

def dice_coeff(a: str, b: str, n: int = 2) -> float:
    A = set(char_ngrams(a, n)); B = set(char_ngrams(b, n))
    if not A and not B: return 1.0
    return 2 * len(A & B) / (len(A) + len(B))

def resolve_unique_course(q: str) -> Tuple[Dict[str, Any], List[Tuple[str, float, int]]]:
    phrases = extract_title_phrases_from_question(q) or [q]
    # 子字串命中 → 命中一筆視為唯一；多筆但第一名領先>=0.5 也通過
    contains_scores = []
    for idx, title in enumerate(COURSE_TITLES):
        tnorm = normalize_text(title)
        score = 0.0
        for p in phrases:
            pnorm = normalize_text(p)
            if not pnorm: continue
            if pnorm in tnorm or tnorm in pnorm:
                score += max(1.0, len(pnorm)/4.0)
        if score > 0.0:
            contains_scores.append((title, score, idx))
    if contains_scores:
        contains_scores.sort(key=lambda x: x[1], reverse=True)
        if len(contains_scores) == 1 or (contains_scores[0][1] - contains_scores[1][1] >= 0.5):
            return DATASET[contains_scores[0][2]], contains_scores[:5]
        else:
            return None, contains_scores[:5]
    # 沒子字串 → Dice
    scores = []
    for idx, title in enumerate(COURSE_TITLES):
        s_max = 0.0
        for p in phrases:
            s_max = max(s_max, dice_coeff(p, title, n=2))
        scores.append((title, s_max, idx))
    scores.sort(key=lambda x: x[1], reverse=True)
    if scores and (scores[0][1] >= 0.58 and (len(scores)==1 or scores[0][1]-scores[1][1] >= 0.08)):
        return DATASET[scores[0][2]], scores[:5]
    else:
        return None, scores[:5]

# ===================== Exact：從 JSON 直接回答 =====================
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

def answer_from_course(q: str, c: Dict[str, Any]) -> str:
    want_textbook = ("教科書" in q) or ("教材" in q) or ("textbook" in q.lower())
    want_credits = ("學分" in q)
    want_teacher = any(k in q for k in ["老師","授課教師"])
    want_midweek = any(k in q for k in ["期中","midterm","Midterm"])
    want_finalweek = any(k in q for k in ["期末","final","Final"])
    want_progress = any(k in q for k in ["週次","進度","syllabus"])
    want_usr = any(k in q for k in ["USR","大學社會責任","關聯性"])
    want_sdgs = any(k in q for k in ["SDGs","永續發展目標"])

    lines = [f"課程名稱：{c['課程名稱']}（{c['學年度']}／{c['學期']}）"]
    lines.append(f"開課班級：{c['開課班級']}　授課教師：{c['授課教師']}　學分數：{c['學分數']}　必選修：{c['必選修']}")

    if want_textbook: lines += ["教科書（全部列出）：", format_textbooks_block(c)]
    if want_credits: lines.append(f"學分數：{c['學分數']}")
    if want_teacher: lines.append(f"授課教師：{c['授課教師']}")
    if want_midweek: lines.append("期中考週次：" + (("、".join(map(str, c.get("mid_weeks", [])))) or "未標註"))
    if want_finalweek: lines.append("期末考週次：" + (("、".join(map(str, c.get("final_weeks", [])))) or "未標註"))
    if want_usr: lines.append(f"大學社會責任(USR)關聯性：{c.get('大學社會責任(USR)關聯性','未標註')}")
    if want_sdgs: lines.append(f"永續發展目標(SDGs)：{c.get('永續發展目標(SDGs)','未標註')}")

    if want_progress and not (want_midweek or want_finalweek):
        lines.append("教學進度表（週次）：")
        lines.append(list_to_lines(c.get("教學進度表", [])))

    if not any([want_textbook,want_credits,want_teacher,want_midweek,want_finalweek,want_progress,want_usr,want_sdgs]):
        lines.append("教學目標："); lines.append(c.get("教學目標","未標註"))
        lines.append("教學綱要："); lines.append(list_to_lines(c.get("教學綱要", [])))
        lines.append("教科書："); lines.append(format_textbooks_block(c))
        lines.append("參考書："); lines.append(list_to_lines(c.get("參考書", [])))
        lines.append("教學進度表："); lines.append(list_to_lines(c.get("教學進度表", [])))
        lines.append("成績評定方式："); lines.append(list_to_lines(c.get("成績評定方式", [])))
        lines.append("課堂要求："); lines.append(list_to_lines(c.get("課堂要求", [])))
        lines.append(f"SDGs：{c.get('永續發展目標(SDGs)','未標註')}")
        lines.append(f"USR 關聯性：{c.get('大學社會責任(USR)關聯性','未標註')}")

    lines.append("如需查詢其他課程，請提供關鍵字（課名/老師/班級）。")
    return "\n".join(lines).strip()

# ===================== 嚴格校園導覽 Prompt（RAG 備用） =====================
def build_campus_guide_prompt(context: str, question: str) -> str:
    rules = """
你是一位「校園課程導覽員」。請嚴格依據〈校務資料〉回答問題，不得臆測或引用外部資料；若在〈校務資料〉找不到答案，請回覆「查無資料」。一律使用繁體中文。

【作答規範（務必遵守）】
1) 只有〈校務資料〉可用；若 context 為空或未命中，直接回覆「查無資料」。
2) 針對課程問題，優先提供：課程名稱、開課班級/年級、授課教師、學分數、必/選修、教科書（若有）、成績評定方式（項目與比例）、教學進度（以週次為主）、USR/SDGs。
3) 命中多門相近課程時，列出最相關前 3 筆，每筆不超過 6 行。
4) 專有名詞與日期請照原文；缺漏以「未標註」表示。
5) 嚴禁捏造未出現在〈校務資料〉中的任何數值、網址或規範。
6) 結尾加上一行：如需查詢其他課程，請提供關鍵字（課名/老師/班級）。
""".strip()
    return f"{rules}\n\n〈問題〉\n{question.strip()}\n\n〈校務資料〉\n{context.strip()}"

# ===================== 檢索與 re-rank（RAG） =====================
def intent_from_question(q: str) -> Dict[str, bool]:
    ql = q.lower()
    return {
        "textbook": ("教科書" in q or "教材" in q or "textbook" in ql),
        "credits": ("學分" in q),
        "teacher": ("老師" in q or "授課教師" in q),
        "midweek": ("期中" in q or "midterm" in ql),
        "finalweek": ("期末" in q or "final" in ql),
        "progress": ("週次" in q or "進度" in q or "syllabus" in ql),
        "usr": ("USR" in q or "大學社會責任" in q or "關聯性" in q),
        "sdgs": ("SDGs" in q or "永續發展目標" in q),
    }

def build_keywords(q: str) -> List[str]:
    it = intent_from_question(q)
    core = []
    if it["textbook"]: core += ["教科書","教材","課程基本資料"]
    if it["credits"]: core += ["學分數","學分","課程基本資料"]
    if it["teacher"]: core += ["授課教師","老師"]
    if it["midweek"]: core += ["期中","教學進度表","週次"]
    if it["finalweek"]: core += ["期末","教學進度表","週次"]
    if it["progress"]: core += ["教學進度表","週次"]
    if it["usr"]: core += ["大學社會責任","USR","關聯性"]
    if it["sdgs"]: core += ["永續發展目標","SDGs"]
    core += ["課程名稱","開課班級"]
    seen, out = set(), []
    for x in core:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def score_doc(doc: str, keywords: List[str], q: str) -> int:
    dl = doc.lower()
    it = intent_from_question(q)
    s = 0
    for kw in keywords:
        kl = kw.lower()
        if kl in dl:
            if kw in ["教科書","教材","學分數","學分","授課教師","老師","大學社會責任","USR","永續發展目標","SDGs"]:
                s += 6
            elif kw in ["教學進度表","週次","期中","期末"]:
                s += 4
            else:
                s += 1
    if ("【課程基本資料】" in doc) and (it["textbook"] or it["credits"] or it["teacher"] or it["usr"] or it["sdgs"]):
        s += 4
    return s

# ===================== 主查詢流程（Exact → RAG） =====================
def run(qs: str):
    global EXACT_MODE, STRICT_UNIQUE, DEBUG

    # --------- Exact：先嘗試唯一命中 ----------
    course, cand = resolve_unique_course(qs)
    if DEBUG:
        print(f"[DEBUG] 片語：{extract_title_phrases_from_question(qs)}")
        if cand:
            print("[DEBUG] 候選：", [(t, round(sc,2)) for (t,sc,_) in cand[:5]])

    if EXACT_MODE and course is not None:
        fields = extract_all_fields(course)
        print("[Exact JSON]")
        print(answer_from_course(qs, fields))
        return
    elif STRICT_UNIQUE and course is None and cand:
        print("[需要更明確的課名]")
        for i, (t, sc, _) in enumerate(cand[:5], 1):
            print(f"{i}. {t}（相似度 {sc:.2f}）")
        return

    # --------- RAG：向量檢索 + 嚴格 Prompt ----------
    try:
        qs_emb = ollama_embed(qs, purpose="query")
    except Exception as e:
        print(f"查詢向量化失敗：{e}")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(name=COLLECTION_NAME)
    res = col.query(query_embeddings=[qs_emb], query_texts=[qs], n_results=TOPN_VECTOR)
    docs = res.get("documents", [[]])[0]

    phrases = extract_title_phrases_from_question(qs)
    if phrases:
        pnorms = [normalize_text(p) for p in phrases]
        filtered = [d for d in docs if any(pn in normalize_text(d) for pn in pnorms)]
        if filtered:
            docs = filtered

    keywords = build_keywords(qs)
    docs_sorted = sorted(docs, key=lambda d: score_doc(d, keywords, qs), reverse=True)
    top_docs = docs_sorted[:TOPK_AFTER_RERANK]
    context = "\n\n---\n\n".join(top_docs)
    prompt = build_campus_guide_prompt(context=context, question=qs)
    model = pick_first_available_model(GEN_MODEL_CANDIDATES)
    try:
        ans = ollama_generate(prompt, model=model)
    except Exception as e:
        print(f"生成失敗（模型：{model}）：{e}")
        return

    print(f"[Model: {model}]")
    print(ans or "查無資料")

# ===================== 互動式 CLI =====================
def print_help():
    print("指令：")
    print("  /help              顯示本說明")
    print("  /reload            重新載入 JSON 並重建向量庫")
    print("  /debug on|off      開/關除錯輸出（顯示片語與候選）")
    print("  /exact on|off      命中唯一課名時直接用 JSON（預設 on）")
    print("  /unique on|off     課名不唯一時是否要求縮小（預設 on）")
    print("  /exit              離開互動模式")

def chat_loop():
    global DEBUG, EXACT_MODE, STRICT_UNIQUE
    print("\n=== 校園課程導覽：互動模式（Exact → RAG） ===")
    print("輸入問題即可查詢；輸入 /help 看指令。\n")
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
            print_help(); continue
        if low == "/reload":
            try:
                initial()
                print("[完成] 已重新載入 JSON 並重建向量庫。")
            except Exception as e:
                print(f"[錯誤] 重建失敗：{e}")
            continue
        if low.startswith("/debug"):
            parts = qs.split()
            if len(parts)==2 and parts[1].lower() in ("on","off"):
                DEBUG = (parts[1].lower()=="on")
                print(f"[設定] Debug = {'開啟' if DEBUG else '關閉'}")
            else:
                print("用法：/debug on|off")
            continue
        if low.startswith("/exact"):
            parts = qs.split()
            if len(parts)==2 and parts[1].lower() in ("on","off"):
                EXACT_MODE = (parts[1].lower()=="on")
                print(f"[設定] Exact（JSON 抽取）= {'開啟' if EXACT_MODE else '關閉'}")
            else:
                print("用法：/exact on|off")
            continue
        if low.startswith("/unique"):
            parts = qs.split()
            if len(parts)==2 and parts[1].lower() in ("on","off"):
                STRICT_UNIQUE = (parts[1].lower()=="on")
                print(f"[設定] 嚴格唯一 = {'開啟' if STRICT_UNIQUE else '關閉'}")
            else:
                print("用法：/unique on|off")
            continue

        # 正式查詢
        try:
            run(qs)
        except Exception as e:
            print(f"[錯誤] 查詢失敗：{e}")

# ===================== 入口 =====================
if __name__ == "__main__":
    initial()
    chat_loop()
