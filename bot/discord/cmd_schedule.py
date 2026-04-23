# -*- coding: utf-8 -*-
"""
bot/cmd_schedule.py — 課表匯入與查詢指令
==========================================
包含：
- /upload_schedule (截圖 OCR 辨識 或 JSON 手動輸入)
- /my_schedule (查詢今日或特定日期課表)
- /my_free (查詢空堂)
- /my_credits (查詢學分摘要)

OCR 引擎：Gemini 3.1 Pro Vision（structured output）
"""

import json
import logging
import discord  # type: ignore
from discord import app_commands  # type: ignore
from datetime import datetime, timezone, timedelta

from bot import tree, logger
from bot.discord.ui_utils import SafeView, safe_respond, safe_defer
from tools.schedule_manager import (
    save_schedule, get_schedule,
    query_day_schedule, query_free_periods, query_credit_summary,
    DAY_NAMES,
)


# =========================================================================
# 📅 課表匯入 Modal（JSON Fallback）
# =========================================================================

class ScheduleModal(discord.ui.Modal, title='📅 匯入課表 (貼上 JSON)'):
    json_input = discord.ui.TextInput(
        label='貼上課表 JSON',
        style=discord.TextStyle.long,
        placeholder='{"academic_year":"114","semester":"2","courses":[...]}',
        required=True,
        max_length=4000,
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        discord_id = str(interaction.user.id)

        try:
            schedule_data = json.loads(self.json_input.value.strip())

            # 支援兩種格式：{"schedule": {...}} 或直接 {"courses": [...]}
            if "schedule" in schedule_data:
                schedule_data = schedule_data["schedule"]

            if "courses" not in schedule_data:
                await interaction.followup.send(
                    "❌ JSON 格式錯誤：找不到 `courses` 欄位。\n"
                    '📋 格式：`{"academic_year":"114","semester":"2","courses":[...]}`',
                    ephemeral=True,
                )
                return

            courses = schedule_data["courses"]
            for i, c in enumerate(courses):
                if "name" not in c or "day" not in c or "periods" not in c:
                    await interaction.followup.send(
                        f"❌ 第 {i+1} 門課缺少必要欄位（需要：`name`, `day`, `periods`）",
                        ephemeral=True,
                    )
                    return

            success = save_schedule(discord_id, schedule_data)
            if success:
                total_credits = sum(c.get("credits", 0) for c in courses)
                embed = _build_schedule_success_embed(schedule_data, courses, total_credits)
                await interaction.followup.send(embed=embed, ephemeral=True)
            else:
                await interaction.followup.send(
                    "❌ 儲存失敗！您可能還沒使用 `/identity_login` 註冊。", ephemeral=True
                )

        except json.JSONDecodeError as e:
            await interaction.followup.send(f"❌ JSON 解析失敗：{e}", ephemeral=True)
        except Exception as e:
            logger.error(f"課表匯入錯誤: {e}")
            await interaction.followup.send(f"❌ 匯入失敗：{e}", ephemeral=True)


# =========================================================================
# ✅ 確認儲存 View（OCR 辨識結果確認）
# =========================================================================

class ScheduleConfirmView(SafeView):
    """OCR 辨識完成後的確認/取消 UI"""
    def __init__(self, discord_id: str, schedule_data: dict):
        super().__init__(owner_id=discord_id)
        self.discord_id = discord_id
        self.schedule_data = schedule_data

    @discord.ui.button(label="✅ 確認儲存", style=discord.ButtonStyle.green, emoji="💾")
    async def confirm_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 已處理過，請勿重複操作。")
            return
        await self.mark_handled(interaction)
        await safe_defer(interaction)
        success = save_schedule(self.discord_id, self.schedule_data)
        if success:
            courses = self.schedule_data.get("courses", [])
            total_credits = sum(c.get("credits", 0) for c in courses)
            embed = _build_schedule_success_embed(self.schedule_data, courses, total_credits)
            await safe_respond(interaction, embed=embed)
        else:
            await safe_respond(interaction, "❌ 儲存失敗！請確認您已使用 `/identity_login` 註冊。")

    @discord.ui.button(label="❌ 取消重傳", style=discord.ButtonStyle.danger, emoji="🔄")
    async def cancel_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 已處理過。")
            return
        await self.mark_handled(interaction)
        await safe_respond(interaction, "📷 已取消！請重新上傳正確的課表截圖。")

    @discord.ui.button(label="📝 改用 JSON 手動輸入", style=discord.ButtonStyle.secondary)
    async def json_fallback_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 已處理過。")
            return
        await self.mark_handled(interaction)
        await interaction.response.send_modal(ScheduleModal())


# =========================================================================
# 🛠️ 輔助函式
# =========================================================================

def _build_schedule_success_embed(schedule_data: dict, courses: list, total_credits: int) -> discord.Embed:
    """建構課表匯入成功的 Embed"""
    embed = discord.Embed(
        title="✅ 課表匯入成功！",
        description=(
            f"📚 **{len(courses)}** 門課 · **{total_credits}** 學分\n"
            f"🗓️ {schedule_data.get('academic_year', '?')}學年度 第{schedule_data.get('semester', '?')}學期"
        ),
        color=discord.Color.green(),
        timestamp=discord.utils.utcnow(),
    )
    for day in range(1, 6):
        day_courses = [c for c in courses if c.get("day") == day]
        if day_courses:
            day_text = "\n".join([
                f"• {c['name']} (第{c['periods'][0]}-{c['periods'][-1]}節)"
                for c in day_courses
            ])
            embed.add_field(name=DAY_NAMES[day], value=day_text, inline=True)
        else:
            embed.add_field(name=DAY_NAMES[day], value="無課 🎉", inline=True)
    embed.set_footer(text="💡 現在可以問我「我星期三有什麼課？」之類的問題了！")
    return embed


def _build_schedule_preview_embed(schedule_data: dict) -> discord.Embed:
    """建構 OCR 辨識結果預覽的 Embed"""
    courses = schedule_data.get("courses", [])
    total_credits = sum(c.get("credits", 0) for c in courses)

    embed = discord.Embed(
        title="🔍 課表 OCR 辨識結果預覽",
        description=(
            f"📚 辨識到 **{len(courses)}** 門課 · 預估 **{total_credits}** 學分\n"
            f"🗓️ {schedule_data.get('academic_year', '?')}學年度 第{schedule_data.get('semester', '?')}學期\n\n"
            f"⚠️ **請仔細確認以下資料是否正確，確認後按「✅ 確認儲存」**"
        ),
        color=discord.Color.gold(),
        timestamp=discord.utils.utcnow(),
    )

    for day in range(1, 8):
        day_courses = [c for c in courses if c.get("day") == day]
        if day_courses:
            day_text = "\n".join([
                f"• **{c['name']}**\n"
                f"  👨‍🏫 {c.get('instructor', '?')} | 📍 {c.get('room', '?')}\n"
                f"  ⏰ 第{c['periods'][0]}-{c['periods'][-1]}節 | {c.get('credits', '?')}學分 ({c.get('type', '?')})"
                for c in day_courses
            ])
            embed.add_field(name=f"📅 {DAY_NAMES.get(day, f'星期{day}')}", value=day_text, inline=False)

    embed.set_footer(text="🤖 由 Gemini 3.1 Pro Vision 辨識 | 若有錯誤請取消重傳或改用 JSON")
    return embed


# =========================================================================
# 📅 /upload_schedule — 匯入課表（截圖 OCR 或 JSON）
# =========================================================================

@tree.command(name="upload_schedule", description="📅 匯入課表（上傳截圖自動辨識或手動貼 JSON）")
@app_commands.describe(
    schedule_image="課表截圖（PNG/JPG，從選課系統截圖）",
)
async def slash_upload_schedule(
    interaction: discord.Interaction,
    schedule_image: discord.Attachment | None = None,
):
    discord_id = str(interaction.user.id)

    # ── 模式 1：截圖 OCR 辨識 ──
    if schedule_image is not None:
        # 檢查檔案類型
        valid_image_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
        if schedule_image.content_type not in valid_image_types:
            await interaction.response.send_message(
                f"❌ 不支援的檔案格式：`{schedule_image.content_type}`\n"
                "📷 請上傳 **PNG / JPG / WEBP** 格式的課表截圖！",
                ephemeral=True,
            )
            return

        # 檢查檔案大小（限制 10MB）
        if schedule_image.size > 10 * 1024 * 1024:
            await interaction.response.send_message(
                "❌ 檔案太大！請上傳 10MB 以下的截圖。", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)

        try:
            # 下載圖片
            image_bytes = await schedule_image.read()
            mime_type = schedule_image.content_type or "image/png"

            # 呼叫 OCR 引擎
            from tools.ocr_engine import ocr_schedule, enrich_schedule_with_course_db
            await interaction.followup.send("⏳ **Gemini 3.1 Pro 正在辨識您的課表截圖...**\n請稍候約 10-20 秒 🔍", ephemeral=True)

            schedule_data = await ocr_schedule(
                image_bytes_list=[image_bytes],
                mime_types=[mime_type],
            )

            if not schedule_data.get("courses"):
                await interaction.followup.send(
                    "⚠️ 辨識完成但沒有找到任何課程！\n"
                    "📷 請確認截圖是否清晰且包含完整的課表表格。\n"
                    "💡 或者使用 `/upload_schedule`（不附檔案）改用 JSON 手動匯入。",
                    ephemeral=True,
                )
                return

            # 🔄 用 RAG 課程資料庫比對，修正學分數和必選修
            import bot as _bot
            if _bot.global_nodes:
                schedule_data = enrich_schedule_with_course_db(schedule_data, _bot.global_nodes)
            else:
                logger.warning("⚠️ RAG 資料庫尚未載入，跳過課程比對增強")

            # 顯示預覽 + 確認按鈕
            preview_embed = _build_schedule_preview_embed(schedule_data)
            confirm_view = ScheduleConfirmView(discord_id, schedule_data)
            await interaction.followup.send(
                embed=preview_embed, view=confirm_view, ephemeral=True
            )

        except ValueError as e:
            await interaction.followup.send(f"❌ 辨識失敗：{e}", ephemeral=True)
        except Exception as e:
            logger.error(f"課表 OCR 錯誤: {e}", exc_info=True)
            await interaction.followup.send(
                f"❌ 發生未預期錯誤：{e}\n💡 請改用 JSON 手動匯入。", ephemeral=True
            )

    # ── 模式 2：JSON 手動匯入（Fallback）──
    else:
        await interaction.response.send_modal(ScheduleModal())


# =========================================================================
# 📋 /my_schedule — 查詢課表
# =========================================================================

@tree.command(name="my_schedule", description="📋 查詢您的課表 (可指定星期幾)")
@app_commands.describe(day="星期幾 (1=一, 2=二, ..., 5=五)，不填=今天")
async def slash_my_schedule(interaction: discord.Interaction, day: int = 0):
    discord_id = str(interaction.user.id)

    if day == 0:
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)
        day = now.isoweekday()
        if day > 5:
            await interaction.response.send_message(
                "🎉 今天是假日，沒有課！\n💡 可以用 `/my_schedule day:1` 查詢星期一的課表", ephemeral=True
            )
            return

    if day < 1 or day > 7:
        await interaction.response.send_message("❌ 請輸入 1-7（1=星期一）", ephemeral=True)
        return

    result = query_day_schedule(discord_id, day)
    await interaction.response.send_message(result, ephemeral=True)


# =========================================================================
# 🕐 /my_free — 查詢空堂
# =========================================================================

@tree.command(name="my_free", description="🕐 查詢您本學期的空堂時段")
async def slash_my_free(interaction: discord.Interaction):
    discord_id = str(interaction.user.id)
    result = query_free_periods(discord_id)
    await interaction.response.send_message(result, ephemeral=True)


# =========================================================================
# 📊 /my_credits — 查詢學分
# =========================================================================

@tree.command(name="my_credits", description="📊 查詢您本學期的修課學分摘要")
async def slash_my_credits(interaction: discord.Interaction):
    discord_id = str(interaction.user.id)
    result = query_credit_summary(discord_id)
    await interaction.response.send_message(result, ephemeral=True)
