# -*- coding: utf-8 -*-
"""
tools/ocr_engine.py — Gemini 3.1 Pro Vision OCR 引擎
=====================================================
使用 Gemini 3.1 Pro 的多模態能力，將課表截圖 / 成績單 PDF
辨識為結構化 JSON，格式完全對齊現有的 token.json schema。

設計原則：
    - temperature = 1.0（Gemini 3 官方建議，推理能力已針對預設值最佳化）
    - responseMimeType = "application/json"（強制 JSON 輸出）
    - responseSchema = 嚴格定義（鎖死在正確格式裡）
    - media_resolution：圖片用 high (1120 tokens)，PDF 用 medium (560 tokens)
    - 圖片在前、指令在後（Gemini Best Practice）
"""

import base64
import json
import logging
import requests
from pathlib import Path

import config

logger = logging.getLogger("discord_bot")


# =============================================================================
# 📋 JSON Schema 定義
# =============================================================================

SCHEDULE_SCHEMA = {
    "type": "object",
    "properties": {
        "academic_year": {
            "type": "string",
            "description": "民國學年度，例如 '114'"
        },
        "semester": {
            "type": "string",
            "description": "學期，'1' 或 '2'"
        },
        "courses": {
            "type": "array",
            "description": "課表中的所有課程清單",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "課程中文名稱"
                    },
                    "name_en": {
                        "type": "string",
                        "description": "課程英文名稱（若圖片中可見）"
                    },
                    "instructor": {
                        "type": "string",
                        "description": "授課教師姓名"
                    },
                    "room": {
                        "type": "string",
                        "description": "上課教室，例如 'E321電腦網路實驗室'"
                    },
                    "day": {
                        "type": "integer",
                        "description": "星期幾，1=一 2=二 3=三 4=四 5=五 6=六 7=日"
                    },
                    "periods": {
                        "type": "array",
                        "description": "上課節次列表，例如 [2,3,4] 表示第2~4節",
                        "items": {"type": "integer"}
                    },
                    "credits": {
                        "type": "integer",
                        "description": "學分數"
                    },
                    "type": {
                        "type": "string",
                        "description": "必修 或 選修",
                        "enum": ["必修", "選修"]
                    }
                },
                "required": ["name", "instructor", "room", "day", "periods", "credits", "type"]
            }
        }
    },
    "required": ["academic_year", "semester", "courses"]
}

TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "student_id": {
            "type": "string",
            "description": "學號"
        },
        "student_name": {
            "type": "string",
            "description": "學生姓名"
        },
        "department": {
            "type": "string",
            "description": "系所名稱，例如 '資訊工程學系'"
        },
        "overall_gpa": {
            "type": "number",
            "description": "學業總平均（歷年）"
        },
        "semesters": {
            "type": "array",
            "description": "每學期的成績資料",
            "items": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "string",
                        "description": "民國學年度，例如 '112'"
                    },
                    "semester": {
                        "type": "string",
                        "description": "學期，'1' 或 '2'"
                    },
                    "gpa": {
                        "type": "number",
                        "description": "該學期操行成績或學業平均"
                    },
                    "courses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "科目名稱"
                                },
                                "credits": {
                                    "type": "integer",
                                    "description": "學分數"
                                },
                                "grade": {
                                    "type": "integer",
                                    "description": "成績分數，若為停修或未完成則填 0"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "類別代碼",
                                    "enum": ["必", "選", "通"]
                                },
                                "status": {
                                    "type": "string",
                                    "description": "修課狀態",
                                    "enum": ["及格", "不及格", "停修", "抵免", "未完成", "修課中"]
                                },
                                "domain": {
                                    "type": "string",
                                    "description": "通識領域（僅通識課填寫），例如 '人文藝術'、'社會科學'、'自然科學'"
                                }
                            },
                            "required": ["name", "credits", "grade", "type", "status"]
                        }
                    }
                },
                "required": ["year", "semester", "courses"]
            }
        },
        "credit_summary": {
            "type": "object",
            "description": "學分摘要",
            "properties": {
                "required_total": {
                    "type": "integer",
                    "description": "畢業應得總學分"
                },
                "required_earned": {
                    "type": "integer",
                    "description": "修習實得總學分"
                },
                "required_remaining": {
                    "type": "integer",
                    "description": "尚缺學分"
                },
                "breakdown": {
                    "type": "object",
                    "description": "各類別學分明細",
                    "properties": {
                        "必修": {
                            "type": "object",
                            "properties": {
                                "required": {"type": "integer", "description": "畢業應得"},
                                "earned": {"type": "integer", "description": "修習實得"},
                                "remaining": {"type": "integer", "description": "尚缺"}
                            },
                            "required": ["required", "earned", "remaining"]
                        },
                        "選修": {
                            "type": "object",
                            "properties": {
                                "required": {"type": "integer", "description": "畢業應得"},
                                "earned": {"type": "integer", "description": "修習實得"},
                                "remaining": {"type": "integer", "description": "尚缺"}
                            },
                            "required": ["required", "earned", "remaining"]
                        },
                        "通識": {
                            "type": "object",
                            "properties": {
                                "required": {"type": "integer", "description": "畢業應得"},
                                "earned": {"type": "integer", "description": "修習實得"},
                                "remaining": {"type": "integer", "description": "尚缺"},
                                "domains": {
                                    "type": "object",
                                    "description": "通識子領域",
                                    "properties": {
                                        "人文藝術": {"type": "object", "properties": {"earned": {"type": "integer"}}, "required": ["earned"]},
                                        "社會科學": {"type": "object", "properties": {"earned": {"type": "integer"}}, "required": ["earned"]},
                                        "自然科學": {"type": "object", "properties": {"earned": {"type": "integer"}}, "required": ["earned"]}
                                    }
                                }
                            },
                            "required": ["required", "earned", "remaining"]
                        }
                    }
                }
            },
            "required": ["required_total", "required_earned", "required_remaining", "breakdown"]
        }
    },
    "required": ["student_id", "overall_gpa", "semesters", "credit_summary"]
}


# =============================================================================
# 📸 OCR Prompt 設計
# =============================================================================

SCHEDULE_OCR_PROMPT = """你是一個專門辨識台灣國立金門大學（NQU）課表截圖的 OCR 提取專家。

【截圖格式說明】
這是一個「星期 × 節次」的表格，結構如下：
- 橫軸（欄）：一(Monday) / 二(Tuesday) / 三(Wednesday) / 四(Thursday) / 五(Friday) / 六(Saturday) / 日(Sunday)
- 縱軸（列）：第1節~第9節，每一格對應一個時段

【NQU 標準時段對照表】
第1節 0810-0900  第2節 0910-1000  第3節 1010-1100
第4節 1110-1200  第5節 1330-1420  第6節 1430-1520
第7節 1530-1620  第8節 1630-1720  第9節 1730-1820

【每格內容結構】
每個有課的格子包含以下資訊（由上到下）：
1. 課程中文名稱
2. 課程英文名稱
3. 授課教師
4. 教室位置

【你的任務】
1. 精確辨識截圖中每一個有課的格子
2. 相同課程如果出現在「同一天的連續節次」（例如第2、3、4節都是同一門課），必須合併為一筆，periods 填入 [2, 3, 4]
3. 學分數 = 該課程佔的連續節數（例如 3 節 = 3 學分）
4. 根據學生的課表內容判斷 academic_year 和 semester（如果截圖中沒有明確標示，請根據當前時間推斷）
5. type 欄位：若無法從截圖判斷必修或選修，預設填 "選修"

【特別注意】
- 教師名字可能有多位（用逗號分隔），請完整保留
- 教室通常格式為「代號+名稱」，如 E321電腦網路實驗室、324企管系專業教室
- 如果某門課出現在不同天（例如星期一和星期三都有），需要拆成兩筆記錄（不同 day）
- 同一天同一門課連續的多個節次必須合併為一筆

請精確辨識並輸出 JSON。"""


TRANSCRIPT_OCR_PROMPT = """你是一個專門辨識台灣國立金門大學（NQU）歷年成績單的 OCR 提取專家。

【成績單格式說明】
這是一份橫向排列的歷年成績表（⚠️ 可能包含多張圖片或多頁，請務必處理「所有」頁面！）：
- 標題區：包含學號、姓名、身分證字號、系所
- 學年區：分為「第一學年(XXX)」、「第二學年(YYY)」等（括號內為民國學年度，依照學生入學年度不同而異），橫跨整份成績單。
- 每學年分為第一學期和第二學期兩欄
- 每門課程一列，包含：類別(必/選/通)、科目名稱、學分、成績

【成績單欄位說明】
- 類別欄的代碼：必=必修、選=選修、通=通識
- 成績欄：數字分數（60 及格）
- 備註欄的代碼：
  - 停=停修、免=免修、抵=抵免
  - /=選修學年課未修
  - 如果成績欄只有分數沒有備註 → 根據分數判斷及格或不及格
  - 備註「未」表示未完成（修課中）
- 底部統計區：
  - 學業成績：該學期平均
  - 實得學分：該學期獲得學分數
  - 累計學分：到該學期為止的累計
  - 操行成績：該學期操行分數

【你的任務】
1. 精確辨識學號、姓名、系所
2. 逐圖片、逐學期、逐課程提取：科目名稱、學分、成績、類別、修課狀態。⚠️ 非常重要：你必須完整掃描「所有」頁面與圖片，不可只提取第一個學期就停下來！如果成績單有多個學期，請保證「全部」都輸出到 JSON 的 semesters 陣列中！
3. 修課狀態判斷規則：
   - 成績 >= 60 且無特殊備註 → "及格"
   - 成績 < 60 → "不及格"
   - 備註為「停」→ "停修"
   - 備註為「抵」或「免」→ "抵免"
   - 備註為「未」或成績欄空白但課程存在 → "未完成"（表示修課中）
4. 通識課需判斷領域（人文藝術/社會科學/自然科學），若無法判斷可留空
5. 底部摘要區：提取畢業應得、修習實得的各類別學分

【學分摘要提取規則】
成績單底部有一個表格，包含：
- 總學分：畢業應得 vs 修習實得
- 共同必修：畢業應得 vs 修習實得
- 通識：畢業應得 vs 修習實得
- 院必修：畢業應得 vs 修習實得
- 專業必修：畢業應得 vs 修習實得
- 專業選修：畢業應得 vs 修習實得

請將「共同必修 + 院必修 + 專業必修」歸類為 breakdown 中的「必修」；
「專業選修」歸類為「選修」；「通識」歸類為「通識」。

【特別注意】
- 學業總平均在成績單的頭部資訊區
- 同一門課可能有中英文名稱，只需保留中文名
- 課程名稱前的 ◎ 符號表示與入學課程規劃表名稱不符，忽略此符號
- 成績若顯示為「-」表示該學期無此成績，不要建立記錄
- 最後一個學期如果成績欄都是空的或標註「未」，表示正在修課中

請精確辨識並輸出 JSON。"""


# =============================================================================
# 🚀 OCR 核心函式
# =============================================================================

async def ocr_schedule(image_bytes_list: list[bytes], mime_types: list[str] | None = None) -> dict:
    """
    課表 OCR：將課表截圖辨識為結構化 JSON。

    Args:
        image_bytes_list: 課表截圖的 bytes 列表（通常只有 1 張）
        mime_types: 每張圖的 MIME 類型，預設 image/png

    Returns:
        結構化課表 JSON（格式同 schedule_manager.save_schedule 的 input）

    Raises:
        ValueError: 辨識失敗或格式錯誤
    """
    if not image_bytes_list:
        raise ValueError("未提供任何課表圖片")

    if mime_types is None:
        mime_types = ["image/png"] * len(image_bytes_list)

    # 組裝 multimodal parts（圖片在前、指令在後）
    parts = []
    for img_bytes, mime in zip(image_bytes_list, mime_types):
        parts.append({
            "inlineData": {
                "mimeType": mime,
                "data": base64.b64encode(img_bytes).decode("utf-8"),
            }
        })
    parts.append({"text": SCHEDULE_OCR_PROMPT})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": config.GEMINI_PRO_MAX_TOKENS,
            "responseMimeType": "application/json",
            "responseSchema": SCHEDULE_SCHEMA,
            "mediaResolution": "media_resolution_high",
            "thinkingConfig": {
                "thinkingLevel": "high"
            }
        }
    }

    logger.info(f"📷 課表 OCR | 送出 {len(image_bytes_list)} 張圖片至 Gemini 3.1 Pro...")

    try:
        import asyncio
        import re

        response = await asyncio.to_thread(
            requests.post,
            config.GEMINI_API_URL,
            json=payload,
            timeout=config.GEMINI_OCR_TIMEOUT,
        )
        response.raise_for_status()
        
        result_text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        # 使用 json_repair 強力修復 LLM 各種奇葩格式
        import json_repair
        try:
            result = json_repair.loads(result_text)
            if not isinstance(result, dict):
                raise ValueError("修復後的 JSON 不是 dict 結構")
        except Exception as e:
            try:
                with open("ocr_schedule_error.json.txt", "w", encoding="utf-8") as err_f:
                    err_f.write(result_text)
            except Exception:
                pass
            logger.error(f"❌ JSON 清理後仍解析失敗: {e}\n已將原始輸出存至 ocr_schedule_error.json.txt")
            raise ValueError(f"JSON 結構異常無法修復：{e}")

        logger.info(f"✅ 課表 OCR 完成 | 辨識到 {len(result.get('courses', []))} 門課")
        return result

    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response else str(e)
        logger.error(f"❌ 課表 OCR API 錯誤: {e} | {error_body[:500]}")
        raise ValueError(f"Gemini API 錯誤：{e}")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"❌ 課表 OCR 解析錯誤: {e}")
        raise ValueError(f"辨識結果解析失敗：{e}")
    except Exception as e:
        logger.error(f"❌ 課表 OCR 未預期錯誤: {e}")
        raise ValueError(f"辨識失敗：{e}")


def enrich_schedule_with_course_db(schedule_data: dict, all_nodes) -> dict:
    """
    用 RAG 課程資料庫比對 OCR 結果，修正學分數和必選修。

    OCR 從截圖只能看到：課名、老師、教室、節次
    看不到的（只能猜的）：學分數、必修/選修

    解法：拿 OCR 辨識出的課程名稱 + 老師名，去 RAG nodes 比對，
    找到對應課程後用資料庫裡的真實值覆蓋 OCR 的猜測值。

    Args:
        schedule_data: OCR 辨識出的課表 dict（含 courses 列表）
        all_nodes: RAG 資料庫中的所有 TextNode（bot.global_nodes）

    Returns:
        增強後的 schedule_data（原地修改並回傳）
    """
    if not all_nodes or not schedule_data.get("courses"):
        return schedule_data

    # ── 建立課程查找索引：{ 課名: { credits, type, teacher, ... } } ──
    course_db = {}
    for node in all_nodes:
        meta = node.metadata
        cname = meta.get("course_name", "")
        section = meta.get("section", "")

        # 只取 basic_info section（每門課的基本資訊）避免重複
        if not cname or section != "basic_info":
            continue

        credits_val = meta.get("credits", "")
        req_type = meta.get("required_or_elective", "")
        teacher = meta.get("teacher", "")

        # 標準化必選修
        if "必" in req_type:
            normalized_type = "必修"
        elif "選" in req_type:
            normalized_type = "選修"
        else:
            normalized_type = ""

        # 學分轉整數
        try:
            credits_int = int(str(credits_val).strip())
        except (ValueError, TypeError):
            credits_int = 0

        # 存入索引（同名課可能有多筆，用 teacher 區分）
        if cname not in course_db:
            course_db[cname] = []
        course_db[cname].append({
            "credits": credits_int,
            "type": normalized_type,
            "teacher": teacher,
        })

    logger.info(f"📚 課程資料庫索引：{len(course_db)} 門課可供比對")

    # ── 逐一比對 OCR 辨識的課程 ──
    matched = 0
    for course in schedule_data["courses"]:
        ocr_name = course.get("name", "")
        ocr_teacher = course.get("instructor", "")

        # 策略 1：精確名稱比對
        candidates = course_db.get(ocr_name)

        # 策略 2：模糊比對（OCR 可能多/少一個字）
        if not candidates:
            for db_name in course_db:
                # 子字串匹配（OCR 名字是 DB 名字的子字串，或反過來）
                if len(ocr_name) >= 3 and (ocr_name in db_name or db_name in ocr_name):
                    candidates = course_db[db_name]
                    break
                # 去括號比對：「程式設計(一)」vs「程式設計」
                ocr_clean = ocr_name.replace("(", "（").replace(")", "）")
                db_clean = db_name.replace("(", "（").replace(")", "）")
                if ocr_clean.split("（")[0] == db_clean.split("（")[0] and len(ocr_clean.split("（")[0]) >= 3:
                    candidates = course_db[db_name]
                    break

        if not candidates:
            logger.debug(f"  ⚠️ 無法比對：{ocr_name}")
            continue

        # 若有多筆同名課，嘗試用老師名來精確匹配
        best = candidates[0]
        if len(candidates) > 1 and ocr_teacher:
            for c in candidates:
                if ocr_teacher in c["teacher"] or c["teacher"] in ocr_teacher:
                    best = c
                    break

        # 覆蓋 OCR 猜的值
        old_credits = course["credits"]
        old_type = course["type"]

        if best["credits"] > 0:
            course["credits"] = best["credits"]
        if best["type"]:
            course["type"] = best["type"]

        if old_credits != course["credits"] or old_type != course["type"]:
            logger.info(
                f"  🔄 修正 [{ocr_name}]：{old_credits}學分→{course['credits']}學分, "
                f"{old_type}→{course['type']}（老師：{best['teacher']}）"
            )
            matched += 1

    logger.info(f"✅ 課表增強完成：{matched}/{len(schedule_data['courses'])} 門課已從資料庫修正")
    return schedule_data


async def ocr_transcript(
    file_bytes_list: list[bytes],
    mime_types: list[str] | None = None,
    user_profile: dict | None = None,
) -> dict:
    """
    成績單 OCR：將成績單 PDF/截圖辨識為結構化 JSON。

    支援：
    - 單一 PDF 檔案（Gemini 原生支援 PDF inline）
    - 多張截圖（自動拼接所有頁面）

    Args:
        file_bytes_list: PDF 或圖片的 bytes 列表
        mime_types: 每個檔案的 MIME 類型
        user_profile: 使用者身分（可選，用於交叉驗證）

    Returns:
        結構化成績單 JSON（格式同 transcript_manager.save_transcript 的 input）

    Raises:
        ValueError: 辨識失敗或格式錯誤
    """
    if not file_bytes_list:
        raise ValueError("未提供任何成績單檔案")

    if mime_types is None:
        mime_types = ["application/pdf"] * len(file_bytes_list)

    # 🚀 優化：PDF 轉高畫質 PNG（解決 Gemini 內建 PDF 解析度太低導致密集表格幻覺的問題）
    processed_bytes_list = []
    processed_mime_types = []
    
    for file_bytes, mime in zip(file_bytes_list, mime_types):
        if mime == "application/pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open("pdf", file_bytes)
                logger.info(f"📄 偵測到 PDF，使用 PyMuPDF 將 {len(doc)} 頁轉為 PNG (縮放 2.0x) 以確保完美辨識...")
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # 放大 2.0 倍 (約 144 DPI)，剛好落在 Gemini 最佳辨識範圍，並強制白底
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    processed_bytes_list.append(pix.tobytes("png"))
                    processed_mime_types.append("image/png")
            except ImportError:
                logger.warning("⚠️ 未安裝 PyMuPDF (fitz)，將使用效能較差的原生 PDF 辨識模式")
                processed_bytes_list.append(file_bytes)
                processed_mime_types.append(mime)
            except Exception as e:
                logger.error(f"❌ PDF 轉檔圖片失敗，退回原生模式：{e}")
                processed_bytes_list.append(file_bytes)
                processed_mime_types.append(mime)
        else:
            processed_bytes_list.append(file_bytes)
            processed_mime_types.append(mime)

    # 組裝 multimodal parts
    parts = []
    for p_bytes, p_mime in zip(processed_bytes_list, processed_mime_types):
        parts.append({
            "inlineData": {
                "mimeType": p_mime,
                "data": base64.b64encode(p_bytes).decode("utf-8"),
            }
        })

    # 若有 user_profile 就加入額外上下文，讓模型可以交叉驗證
    extra_context = ""
    if user_profile:
        dept = user_profile.get("department", "")
        grade = user_profile.get("grade", "")
        if dept:
            extra_context = f"\n【學生身分參考】系所：{dept}、年級：{grade}\n"

    parts.append({"text": TRANSCRIPT_OCR_PROMPT + extra_context})

    # PDF 用 medium、圖片用 high（官方建議）。經過 PyMuPDF 轉換後此處多半為 False (image/png)
    is_pdf = any("pdf" in m for m in processed_mime_types)
    media_res = "media_resolution_medium" if is_pdf else "media_resolution_high"

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": config.GEMINI_PRO_MAX_TOKENS,
            "responseMimeType": "application/json",
            "responseSchema": TRANSCRIPT_SCHEMA,
            "mediaResolution": media_res,
            "thinkingConfig": {
                "thinkingLevel": "high"
            }
        }
    }

    logger.info(f"📊 成績單 OCR | 送出 {len(file_bytes_list)} 個檔案至 Gemini 3.1 Pro (media_resolution={media_res})...")

    try:
        import asyncio
        import re

        # 1. 使用 asyncio.to_thread 避免長達 90 秒的同步請求卡死 Discord Heartbeat
        response = await asyncio.to_thread(
            requests.post,
            config.GEMINI_API_URL,
            json=payload,
            timeout=config.GEMINI_OCR_TIMEOUT,
        )
        response.raise_for_status()
        
        result_text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        # 使用 json_repair 強力修復 LLM 各種奇葩格式 (尾隨逗號、漏引號、換行壞掉等)
        import json_repair
        try:
            result = json_repair.loads(result_text)
            if not isinstance(result, dict):
                raise ValueError("修復後的 JSON 不是 dict 結構")
        except Exception as e:
            # 發生極端解析錯誤時，將原始 JSON 寫入檔案方便除錯
            try:
                with open("ocr_transcript_error.json.txt", "w", encoding="utf-8") as err_f:
                    err_f.write(result_text)
            except Exception:
                pass
            logger.error(f"❌ JSON 清理後仍解析失敗: {e}\n已將原始輸出存至 ocr_transcript_error.json.txt")
            raise ValueError(f"JSON 結構異常無法修復：{e}")

        semesters = result.get("semesters", [])
        total_courses = sum(len(s.get("courses", [])) for s in semesters)
        logger.info(f"✅ 成績單 OCR 完成 | {len(semesters)} 學期, {total_courses} 門課")

        # 後處理：補充 required_remaining
        cs = result.get("credit_summary", {})
        if "required_remaining" not in cs or cs["required_remaining"] == 0:
            total = cs.get("required_total", 130)
            earned = cs.get("required_earned", 0)
            cs["required_remaining"] = max(0, total - earned)

        # 後處理：確保 breakdown 的 remaining 正確
        for cat_name, cat_data in cs.get("breakdown", {}).items():
            if isinstance(cat_data, dict) and "remaining" not in cat_data:
                cat_data["remaining"] = max(0, cat_data.get("required", 0) - cat_data.get("earned", 0))

        return result

    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response else str(e)
        logger.error(f"❌ 成績單 OCR API 錯誤: {e} | {error_body[:500]}")
        raise ValueError(f"Gemini API 錯誤：{e}")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"❌ 成績單 OCR 解析錯誤: {e}")
        raise ValueError(f"辨識結果解析失敗：{e}")
    except Exception as e:
        logger.error(f"❌ 成績單 OCR 未預期錯誤: {e}")
        raise ValueError(f"辨識失敗：{e}")
