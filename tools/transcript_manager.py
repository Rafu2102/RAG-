# -*- coding: utf-8 -*-
"""
tools/transcript_manager.py — 歷年成績單資料管理
================================================
管理學生歷年成績的 JSON 儲存、讀取與查詢。

存放位置：tools/data/tokens/{discord_id}_token.json 的 "transcript" 欄位
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("discord_bot")

TOKEN_DIR = Path(__file__).parent / "data" / "discord_tokens"

# ── 雙軌路由：tg_ 前綴 → telegram_tokens/ ──
_TG_TOKEN_DIR = Path(__file__).parent / "data" / "telegram_tokens"

def _get_user_token_path(user_id: str) -> Path:
    if user_id.startswith("tg_"):
        return _TG_TOKEN_DIR / f"{user_id[3:]}_token.json"
    return TOKEN_DIR / f"{user_id}_token.json"


def _load_user_data(discord_id: str) -> dict | None:
    path = _get_user_token_path(discord_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_user_data(discord_id: str, data: dict):
    path = _get_user_token_path(discord_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _load_graduation_rules() -> dict | None:
    rules_dir = Path(__file__).parent.parent / "data" / "rules"
    if rules_dir.exists():
        for path in rules_dir.glob("*.json"):
            if "graduation" in path.name.lower() or "畢業" in path.name:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
    return None


# =========================================================================
# 📊 成績單存取
# =========================================================================

def save_transcript(discord_id: str, transcript_data: dict) -> bool:
    """
    儲存成績單資料到使用者 token 的 transcript 欄位。
    支援有/無 "transcript" 包裝的 JSON。
    """
    user_data = _load_user_data(discord_id)
    if user_data is None:
        logger.error(f"找不到使用者 {discord_id} 的 token 檔案")
        return False

    # 支援兩種格式：{"transcript": {...}} 或直接 {...}
    if "transcript" in transcript_data:
        transcript = transcript_data["transcript"]
    else:
        transcript = transcript_data

    # 更新時間戳
    tz = timezone(timedelta(hours=8))
    transcript["updated_at"] = datetime.now(tz).isoformat()

    user_data["transcript"] = transcript
    _save_user_data(discord_id, user_data)

    semesters = len(transcript.get("semesters", []))
    earned = transcript.get("credit_summary", {}).get("required_earned", 0)
    logger.info(f"📊 成績單儲存 | 使用者={discord_id} | {semesters} 學期 {earned} 已修學分")
    return True


def get_transcript(discord_id: str) -> dict | None:
    """取得使用者的成績單資料"""
    user_data = _load_user_data(discord_id)
    if user_data is None:
        return None
    return user_data.get("transcript")


# =========================================================================
# 🔍 成績單查詢工具
# =========================================================================

def query_credit_progress(discord_id: str) -> str:
    """查詢學分進度（與畢業標準比對）"""
    transcript = get_transcript(discord_id)
    if not transcript:
        return "❌ 您還沒有匯入成績單資料。請先使用 `/upload_transcript` 上傳。"

    summary = transcript.get("credit_summary", {})
    breakdown = summary.get("breakdown", {})

    total_req = summary.get("required_total", 130)
    earned = summary.get("required_earned", 0)
    remaining = summary.get("required_remaining", total_req - earned)

    lines = [f"📊 **畢業學分進度** — 已修 {earned}/{total_req} 學分（還差 {remaining}）"]

    # 抓取本學期修課中的課程
    ongoing_courses = {}
    semesters = transcript.get("semesters", [])
    
    # 預載畢業門檻標準備用，以便判斷外系必修
    rules = _load_graduation_rules()
    dept_req_courses = set()
    if rules:
        cats = rules.get("categories", {})
        for rc in ["共同必修", "院必修", "系必修"]:
            for req_c in cats.get(rc, {}).get("courses", []):
                if isinstance(req_c, dict) and "name" in req_c:
                    dept_req_courses.add(req_c["name"].replace(" ", ""))
                elif isinstance(req_c, str):
                    dept_req_courses.add(req_c.replace(" ", ""))
        dept_req_courses.update(["體育", "國文", "英文", "服務教育"])
    
    if semesters:
        latest_sem = semesters[-1] # 最後一個通常是最近的學期
        for c in latest_sem.get("courses", []):
            if c.get("status") in ("未完成", "修課中"):
                # 將 "必", "選", "通" 對應到 breakdown 的 key
                t = c.get("type", "")
                cat_key = {"必": "必修", "選": "選修", "通": "通識"}.get(t, t)
                
                # 強制校正：如果是必修，但不在本系的必修清單內（也就是外系必修），則視為選修
                if cat_key == "必修" and dept_req_courses:
                    cname = c["name"].replace(" ", "")
                    # 模糊匹配 (例如 "國文(一)" in "國文")
                    is_own_req = any((req in cname or cname in req) for req in dept_req_courses)
                    if not is_own_req:
                        cat_key = "選修"
                        # 可選：在後面加個註記提示是外系必修
                        c["name"] = f"{c['name']} (外系必修→算入選修)"
                
                if cat_key not in ongoing_courses:
                    ongoing_courses[cat_key] = []
                ongoing_courses[cat_key].append(f"{c['name']} ({c.get('credits', '?')}學分)")

    # === 新增：動態重算歷年及格的外系必修，將其學分從「必修」轉移至「選修」 ===
    cross_dept_creds = 0
    cross_dept_names = []
    
    if dept_req_courses and semesters:
        for sem in semesters:
            for c in sem.get("courses", []):
                # 只針對已經及格或抵免的課程進行已修學分校正
                if c.get("status") in ("及格", "抵免"):
                    t = c.get("type", "")
                    cat_key = {"必": "必修", "選": "選修", "通": "通識"}.get(t, t)
                    if cat_key == "必修":
                        cname = c["name"].replace(" ", "")
                        is_own_req = any((req in cname or cname in req) for req in dept_req_courses)
                        if not is_own_req:
                            creds = c.get("credits", 0)
                            cross_dept_creds += creds
                            cross_dept_names.append(f"{c['name']}({creds}學分)")

    if cross_dept_creds > 0:
        b_req = breakdown.get("必修", {})
        b_elec = breakdown.get("選修", {})
        
        # 扣除必修已修學分
        if isinstance(b_req, dict) and "earned" in b_req:
            b_req["earned"] = max(0, b_req["earned"] - cross_dept_creds)
            b_req["remaining"] = max(0, b_req.get("required", 51) - b_req["earned"])
            
        # 增加選修已修學分
        if isinstance(b_elec, dict) and "earned" in b_elec:
            b_elec["earned"] += cross_dept_creds
            b_elec["remaining"] = max(0, b_elec.get("required", 49) - b_elec["earned"])
            
    # 各類別
    for cat_name, cat_data in breakdown.items():
        if isinstance(cat_data, dict):
            cat_earned = cat_data.get("earned", 0)
            cat_req = cat_data.get("required", 0)
            cat_rem = cat_data.get("remaining", max(0, cat_req - cat_earned))

            if cat_rem <= 0:
                over = cat_earned - cat_req
                over_str = f" (多修 {over} 學分)" if over > 0 else ""
                lines.append(f"  ✅ **{cat_name}**：{cat_earned}/{cat_req} — 已完成{over_str}")
            else:
                lines.append(f"  ⚠️ **{cat_name}**：{cat_earned}/{cat_req} — 還差 {cat_rem} 學分")

            # 顯示本學期修課
            ongoing = ongoing_courses.get(cat_name, [])
            if ongoing:
                lines.append(f"    ⏳ 本學期修課中：{', '.join(ongoing)}")

            # 通識子領域
            if "domains" in cat_data:
                for domain, ddata in cat_data["domains"].items():
                    d_earned = ddata.get("earned", 0)
                    lines.append(f"    • {domain}：已修 {d_earned} 學分")

            # 學期完成度
            if "semesters_completed" in cat_data:
                lines.append(f"    學期：{cat_data['semesters_completed']}")
        else:
            lines.append(f"  📝 **{cat_name}**：{cat_data}")

    # ===== AI 學分策略盤點與推論分析 =====
    advice = []
    
    # 外系必修校正提示
    if cross_dept_names:
        advice.append(f"🔄 **學分自動校正**：偵測到您修了 {len(cross_dept_names)} 門外系必修 ({', '.join(cross_dept_names)})。根據規定，外系必修僅可抵充專業選修學分，系統已自動將其計入「選修」類別。")
    
    # 通識判斷
    ge_earned = breakdown.get("通識", {}).get("earned", 0)
    ge_req = breakdown.get("通識", {}).get("required", 16)
    if ge_earned >= ge_req:
        over = ge_earned - ge_req
        over_txt = f" (您目前已多出 {over} 學分)" if over > 0 else ""
        advice.append(f"🛡️ **通識滿載**：您的通識學分已經達標{over_txt}！若本學期有排通識課，建議退選，將黃金時段留給「專業選修」或「專題」。")
    elif ge_earned < ge_req:
        advice.append(f"💡 **通識警報**：您的通識還差 {ge_req - ge_earned} 學分，請留意【人文/社會/自然】三大領域分配，避免偏廢。")

    # 選修判斷
    el_earned = breakdown.get("選修", {}).get("earned", 0)
    el_req = breakdown.get("選修", {}).get("required", 49)
    
    # 計算本學期正在修的選修學分數
    import re
    el_ongoing_credits = 0
    for c_str in ongoing_courses.get("選修", []):
        m = re.search(r"\((\d+)學分\)", c_str)
        if m:
            el_ongoing_credits += int(m.group(1))
            
    real_remaining_el = max(0, el_req - el_earned - el_ongoing_credits)
            
    if el_earned >= el_req:
         advice.append("🎯 **專業選修達標**：本系選修已過門檻，若繼續修課，超出的學分將可用來補足畢業的 130 總學分數。")
    elif el_earned + el_ongoing_credits >= el_req:
         advice.append(f"🎯 **專業選修預計達標**：加上本學期正在修的 {el_ongoing_credits} 學分，您的專業選修將順利過關！穩住別被當掉就好。")
    elif real_remaining_el > 15:
         advice.append(f"📈 **選修進度提醒**：扣除本學期修課後，您**還差 {real_remaining_el} 學分**選修要拼。建議後續每學期至少安排 3~4 門系內主題選修，加速達標。")
    else:
         advice.append(f"📈 **選修進度提醒**：扣除本學期修課後，您**僅剩 {real_remaining_el} 學分**選修即可達標。再加把勁！")

    # 未完成提示
    if remaining > 0 and remaining < 10:
        advice.append(f"🎉 **畢業在望**：革命即將成功！您只差最後 {remaining} 學分，請專注衝刺剩下的必修與門檻規定。")

    if advice:
        lines.append("\n🤖 **[AI 專屬選課策略與分析]**")
        for a in advice:
            lines.append(a)

    return "\n".join(lines)


def query_failed_courses(discord_id: str) -> str:
    """查詢不及格課程"""
    transcript = get_transcript(discord_id)
    if not transcript:
        return "❌ 您還沒有匯入成績單資料。"

    failed = transcript.get("failed_courses", [])
    if not failed:
        # 從所有學期中找不及格的
        failed_list = []
        for sem in transcript.get("semesters", []):
            for c in sem.get("courses", []):
                if c.get("status") in ("不及格", "停修", "未完成"):
                    failed_list.append({
                        "name": c["name"],
                        "credits": c.get("credits", 0),
                        "semester": f"{sem['year']}-{sem['semester']}",
                        "status": c.get("status", "?"),
                        "grade": c.get("grade"),
                    })
        if not failed_list:
            return "🎉 恭喜！您沒有不及格或未完成的課程！"

        lines = [f"📋 **不及格/未完成課程** — 共 {len(failed_list)} 門"]
        for c in failed_list:
            grade_str = f" ({c['grade']}分)" if c.get("grade") is not None else ""
            lines.append(f"  ❌ **{c['name']}** ({c['credits']}學分) | {c['semester']} | {c['status']}{grade_str}")
        return "\n".join(lines)
    else:
        if not failed:
            return "🎉 恭喜！您沒有不及格的課程！"
        lines = [f"📋 **不及格課程** — 共 {len(failed)} 門"]
        for c in failed:
            lines.append(f"  ❌ {c['name']} ({c['credits']}學分) — {c['semester']}")
        return "\n".join(lines)


def query_gpa(discord_id: str) -> str:
    """查詢歷年 GPA"""
    transcript = get_transcript(discord_id)
    if not transcript:
        return "❌ 您還沒有匯入成績單資料。"

    overall = transcript.get("overall_gpa", "?")
    lines = [f"📊 **歷年總平均**：{overall}"]

    for sem in transcript.get("semesters", []):
        year = sem.get("year", "?")
        semester = sem.get("semester", "?")
        gpa = sem.get("gpa", "?")
        course_count = len(sem.get("courses", []))
        lines.append(f"  {year}學年 第{semester}學期：{gpa} ({course_count}門)")

    return "\n".join(lines)


def get_transcript_context_for_llm(discord_id: str, query: str = "") -> str:
    """
    為 LLM 生成完整的成績單 context。
    不再為了節省 Token 而切割資料，直接將「整體學分分析」進度與「歷年所有修課明細」
    一併傳給具備超大 Context Window 的 Gemini 3.1 Pro，讓 AI 擁有完整的全局視野。
    """
    transcript = get_transcript(discord_id)
    if not transcript:
        return ""

    # 1. 取得 AI 策略分析與學分總覽
    analysis = query_credit_progress(discord_id)
    
    # 2. 展開所有歷年修課明細
    history_lines = ["\n【歷年所有修課明細】"]
    semesters = transcript.get("semesters", [])
    if not semesters:
        history_lines.append("⚠️ 目前查無任何學期的修課明細資料。")
    else:
        for sem in semesters:
            year = sem.get("year", "?")
            semester = sem.get("semester", "?")
            gpa = sem.get("gpa", "?")
            history_lines.append(f"📌 {year}學年度 第{semester}學期 (GPA: {gpa}):")
            for c in sem.get("courses", []):
                c_name = c.get("name", "?")
                c_type = c.get("type", "?")
                c_cred = c.get("credits", "?")
                c_grade = c.get("grade", "?")
                c_status = c.get("status", "?")
                history_lines.append(f"  - {c_name} | {c_type} | {c_cred}學分 | 成績: {c_grade} ({c_status})")
            history_lines.append("") # 留空行分隔

    history_text = "\n".join(history_lines)
    
    # 3. 組合回傳
    return f"【學生的整體學分進度與專屬分析】\n{analysis}\n{history_text}"

