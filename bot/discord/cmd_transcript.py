# -*- coding: utf-8 -*-
"""
bot/cmd_transcript.py — 成績單匯入與查詢指令
==============================================
包含：
- /upload_transcript (PDF OCR 辨識 或 JSON 手動輸入)
- /my_credits_total (查詢畢業學分進度)
- /my_gpa (查詢歷年 GPA)
- /my_failed (查詢不及格/未完成課程)

OCR 引擎：Gemini 3.1 Pro Vision（structured output）
"""

import json
import logging
import discord  # type: ignore
from discord import app_commands  # type: ignore

from bot import tree, logger
from bot.discord.ui_utils import SafeView, safe_respond, safe_defer
from tools.transcript_manager import (
    save_transcript, get_transcript,
    query_credit_progress, query_failed_courses, query_gpa,
)


# =========================================================================
# 📊 成績單匯入 Modal（JSON Fallback）
# =========================================================================

class TranscriptModal(discord.ui.Modal, title='📊 匯入成績單 (貼上 JSON)'):
    json_input = discord.ui.TextInput(
        label='貼上成績單 JSON（可壓縮空白）',
        style=discord.TextStyle.long,
        placeholder='{"transcript":{"student_id":"...","semesters":[...]}}',
        required=True,
        max_length=4000,
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        discord_id = str(interaction.user.id)
        await _process_transcript_json(interaction, discord_id, self.json_input.value.strip())


# =========================================================================
# 📊 共用 JSON 解析邏輯
# =========================================================================

async def _process_transcript_json(interaction: discord.Interaction, discord_id: str, raw_json: str):
    """解析並儲存成績單 JSON（Modal 和附件共用）"""
    try:
        data = json.loads(raw_json)

        # 支援兩種格式
        if "transcript" in data:
            transcript_data = data["transcript"]
        elif "semesters" in data:
            transcript_data = data
        else:
            await interaction.followup.send(
                "❌ JSON 格式錯誤：找不到 `semesters` 欄位。\n"
                '📋 格式：`{"transcript":{"semesters":[...],"credit_summary":{...}}}`',
                ephemeral=True,
            )
            return

        semesters = transcript_data.get("semesters", [])
        if not semesters:
            await interaction.followup.send("❌ 找不到任何學期資料。", ephemeral=True)
            return

        success = save_transcript(discord_id, {"transcript": transcript_data})
        if success:
            embed = _build_transcript_success_embed(transcript_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.followup.send(
                "❌ 儲存失敗！您可能還沒使用 `/identity_login` 註冊。", ephemeral=True
            )

    except json.JSONDecodeError as e:
        await interaction.followup.send(f"❌ JSON 解析失敗：{e}", ephemeral=True)
    except Exception as e:
        logger.error(f"成績單匯入錯誤: {e}")
        await interaction.followup.send(f"❌ 匯入失敗：{e}", ephemeral=True)


# =========================================================================
# ✅ 確認儲存 View（OCR 辨識結果確認）
# =========================================================================

class TranscriptConfirmView(SafeView):
    """OCR 辨識完成後的確認/取消 UI"""
    def __init__(self, discord_id: str, transcript_data: dict):
        super().__init__(owner_id=discord_id)
        self.discord_id = discord_id
        self.transcript_data = transcript_data

    @discord.ui.button(label="✅ 確認儲存", style=discord.ButtonStyle.green, emoji="💾")
    async def confirm_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 已處理過，請勿重複操作。")
            return
        await self.mark_handled(interaction)
        await safe_defer(interaction)
        success = save_transcript(self.discord_id, {"transcript": self.transcript_data})
        if success:
            embed = _build_transcript_success_embed(self.transcript_data)
            await safe_respond(interaction, embed=embed)
        else:
            await safe_respond(interaction, "❌ 儲存失敗！請確認您已使用 `/identity_login` 註冊。")

    @discord.ui.button(label="❌ 取消重傳", style=discord.ButtonStyle.danger, emoji="🔄")
    async def cancel_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 已處理過。")
            return
        await self.mark_handled(interaction)
        await safe_respond(interaction, "📄 已取消！請重新上傳正確的成績單檔案。")

    @discord.ui.button(label="📝 改用 JSON 手動輸入", style=discord.ButtonStyle.secondary)
    async def json_fallback_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 已處理過。")
            return
        await self.mark_handled(interaction)
        await interaction.response.send_modal(TranscriptModal())


# =========================================================================
# 🛠️ 輔助函式
# =========================================================================

def _build_transcript_success_embed(transcript_data: dict) -> discord.Embed:
    """建構成績單匯入成功的 Embed"""
    semesters = transcript_data.get("semesters", [])
    total_courses = sum(len(s.get("courses", [])) for s in semesters)
    earned = transcript_data.get("credit_summary", {}).get("required_earned", "?")
    overall_gpa = transcript_data.get("overall_gpa", "?")

    embed = discord.Embed(
        title="✅ 成績單匯入成功！",
        description=(
            f"📚 **{len(semesters)}** 學期 · **{total_courses}** 門課\n"
            f"📊 已修學分：**{earned}** · 總平均：**{overall_gpa}**"
        ),
        color=discord.Color.green(),
        timestamp=discord.utils.utcnow(),
    )

    summary = transcript_data.get("credit_summary", {})
    breakdown = summary.get("breakdown", {})
    for cat_name, cat_data in breakdown.items():
        if isinstance(cat_data, dict):
            cat_earned = cat_data.get("earned", 0)
            cat_req = cat_data.get("required", 0)
            cat_rem = cat_data.get("remaining", max(0, cat_req - cat_earned))
            status = "✅" if cat_rem == 0 else "⚠️"
            embed.add_field(
                name=f"{status} {cat_name}",
                value=f"{cat_earned}/{cat_req}" + (f" (差{cat_rem})" if cat_rem > 0 else " 已完成"),
                inline=True,
            )

    embed.set_footer(text="💡 現在可以問我「我的必修還差幾學分？」之類的問題了！")
    return embed


def _build_transcript_preview_embed(transcript_data: dict) -> discord.Embed:
    """建構 OCR 辨識結果預覽的 Embed"""
    semesters = transcript_data.get("semesters", [])
    total_courses = sum(len(s.get("courses", [])) for s in semesters)
    overall_gpa = transcript_data.get("overall_gpa", "?")
    student_id = transcript_data.get("student_id", "?")

    embed = discord.Embed(
        title="🔍 成績單 OCR 辨識結果預覽",
        description=(
            f"🆔 學號：**{student_id}**\n"
            f"📚 辨識到 **{len(semesters)}** 學期 · **{total_courses}** 門課\n"
            f"📊 學業總平均：**{overall_gpa}**\n\n"
            f"⚠️ **請仔細確認以下資料是否正確，確認後按「✅ 確認儲存」**"
        ),
        color=discord.Color.gold(),
        timestamp=discord.utils.utcnow(),
    )

    # 每學期顯示摘要（不列出每門課，太長了）
    for sem in semesters:
        year = sem.get("year", "?")
        semester = sem.get("semester", "?")
        courses = sem.get("courses", [])
        gpa = sem.get("gpa", "?")
        total_creds = sum(c.get("credits", 0) for c in courses)

        # 統計及格/不及格
        passed = sum(1 for c in courses if c.get("status") in ("及格", "抵免"))
        failed = sum(1 for c in courses if c.get("status") == "不及格")
        ongoing = sum(1 for c in courses if c.get("status") in ("未完成", "修課中"))
        stopped = sum(1 for c in courses if c.get("status") == "停修")

        status_parts = []
        if passed: status_parts.append(f"✅{passed}")
        if failed: status_parts.append(f"❌{failed}")
        if stopped: status_parts.append(f"🔄{stopped}")
        if ongoing: status_parts.append(f"⏳{ongoing}")
        status_str = " · ".join(status_parts) if status_parts else "—"

        # 列出課程名稱（簡短版）
        course_names = ", ".join([c.get("name", "?")[:8] for c in courses[:6]])
        if len(courses) > 6:
            course_names += f"...等 {len(courses)} 門"

        embed.add_field(
            name=f"📅 {year}-{semester} | {total_creds}學分 | {status_str}",
            value=f"📖 {course_names}\n📊 操行/學業：{gpa}",
            inline=False,
        )

    # 學分摘要
    cs = transcript_data.get("credit_summary", {})
    breakdown = cs.get("breakdown", {})
    summary_parts = []
    for cat_name, cat_data in breakdown.items():
        if isinstance(cat_data, dict):
            summary_parts.append(f"{cat_name} {cat_data.get('earned', 0)}/{cat_data.get('required', 0)}")
    if summary_parts:
        embed.add_field(
            name="📊 學分摘要",
            value=" | ".join(summary_parts),
            inline=False,
        )

    embed.set_footer(text="🤖 由 Gemini 3.1 Pro Vision 辨識 | 若有錯誤請取消重傳或改用 JSON")
    return embed


# =========================================================================
# 📊 /upload_transcript — 匯入成績單（PDF OCR 或 JSON）
# =========================================================================

@tree.command(name="upload_transcript", description="📊 匯入歷年成績單（上傳 PDF 自動辨識或 JSON 手動輸入）")
@app_commands.describe(
    transcript_file="成績單檔案（PDF 自動 OCR 辨識 / JSON 手動匯入）",
)
async def slash_upload_transcript(
    interaction: discord.Interaction,
    transcript_file: discord.Attachment | None = None,
):
    discord_id = str(interaction.user.id)

    if transcript_file is not None:
        filename = transcript_file.filename.lower()
        content_type = transcript_file.content_type or ""

        # ── 判斷檔案類型 ──
        is_json = filename.endswith(".json") or "json" in content_type
        is_pdf = filename.endswith(".pdf") or "pdf" in content_type
        is_image = any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) or content_type.startswith("image/")

        # ── JSON 模式（手動匯入）──
        if is_json:
            await interaction.response.defer(ephemeral=True)
            content = await transcript_file.read()
            await _process_transcript_json(interaction, discord_id, content.decode("utf-8"))
            return

        # ── PDF / 圖片 OCR 模式 ──
        if is_pdf or is_image:
            # 檢查檔案大小（限制 15MB）
            if transcript_file.size > 15 * 1024 * 1024:
                await interaction.response.send_message(
                    "❌ 檔案太大！請上傳 15MB 以下的檔案。", ephemeral=True
                )
                return

            await interaction.response.defer(ephemeral=True)

            try:
                file_bytes = await transcript_file.read()

                if is_pdf:
                    mime_type = "application/pdf"
                elif filename.endswith(".png"):
                    mime_type = "image/png"
                elif filename.endswith(".webp"):
                    mime_type = "image/webp"
                else:
                    mime_type = "image/jpeg"

                # 取得使用者 profile 作為交叉驗證參考
                from tools.auth import get_user_profile
                user_profile = get_user_profile(discord_id)

                # 呼叫 OCR 引擎
                from tools.ocr_engine import ocr_transcript
                await interaction.followup.send(
                    "⏳ **Gemini 3.1 Pro 正在辨識您的成績單...**\n"
                    "📄 PDF 辨識可能需要 15-30 秒，請稍候 🔍",
                    ephemeral=True,
                )

                transcript_data = await ocr_transcript(
                    file_bytes_list=[file_bytes],
                    mime_types=[mime_type],
                    user_profile=user_profile,
                )

                if not transcript_data.get("semesters"):
                    await interaction.followup.send(
                        "⚠️ 辨識完成但沒有找到任何學期資料！\n"
                        "📄 請確認上傳的是「歷年成績表」PDF。\n"
                        "💡 或改用 `/upload_transcript`（附 .json 檔）手動匯入。",
                        ephemeral=True,
                    )
                    return

                # 顯示預覽 + 確認按鈕
                preview_embed = _build_transcript_preview_embed(transcript_data)
                confirm_view = TranscriptConfirmView(discord_id, transcript_data)
                await interaction.followup.send(
                    embed=preview_embed, view=confirm_view, ephemeral=True
                )

            except ValueError as e:
                await interaction.followup.send(f"❌ 辨識失敗：{e}", ephemeral=True)
            except Exception as e:
                logger.error(f"成績單 OCR 錯誤: {e}", exc_info=True)
                await interaction.followup.send(
                    f"❌ 發生未預期錯誤：{e}\n💡 請改用 JSON 手動匯入。", ephemeral=True
                )
            return

        # ── 不支援的格式 ──
        await interaction.response.send_message(
            f"❌ 不支援的檔案格式：`{filename}`\n"
            "📄 請上傳 **PDF**（歷年成績表）或 **JSON** 格式！\n"
            "📷 也支援 **PNG / JPG** 截圖。",
            ephemeral=True,
        )

    # ── 無附件 → Modal 模式（貼 JSON）──
    else:
        await interaction.response.send_modal(TranscriptModal())


# =========================================================================
# 📊 /my_credits_total — 查詢畢業學分進度
# =========================================================================

@tree.command(name="my_credits_total", description="📊 查詢您的畢業學分進度（與畢業標準比對）")
async def slash_my_credits_total(interaction: discord.Interaction):
    discord_id = str(interaction.user.id)
    result = query_credit_progress(discord_id)
    await interaction.response.send_message(result, ephemeral=True)


# =========================================================================
# 📈 /my_gpa — 查詢歷年 GPA
# =========================================================================

@tree.command(name="my_gpa", description="📈 查詢您的歷年成績平均")
async def slash_my_gpa(interaction: discord.Interaction):
    discord_id = str(interaction.user.id)
    result = query_gpa(discord_id)
    await interaction.response.send_message(result, ephemeral=True)


# =========================================================================
# ❌ /my_failed — 查詢不及格課程
# =========================================================================

@tree.command(name="my_failed", description="❌ 查詢您的不及格或未完成課程")
async def slash_my_failed(interaction: discord.Interaction):
    discord_id = str(interaction.user.id)
    result = query_failed_courses(discord_id)
    await interaction.response.send_message(result, ephemeral=True)
