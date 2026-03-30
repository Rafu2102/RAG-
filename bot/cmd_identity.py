# -*- coding: utf-8 -*-
"""
bot/cmd_identity.py — OAuth 授權與身分管理指令
===============================================
包含：
- CalendarAuthModal / CalendarAuthView (OAuth 註冊流程 UI)
- IdentityManageView / DeleteConfirmView (身分管理 UI)
- /identity_login 斜線指令
"""

import logging
import discord  # type: ignore

from bot import tree, DEPT_MAPPING, logger
from bot.ui_utils import SafeView, safe_respond, safe_defer
from tools.auth import get_auth_url, verify_and_save_token, get_user_profile, delete_user_token


# =========================================================================
# 🌟 OAuth 授權與身分註冊介面
# =========================================================================

class CalendarAuthModal(discord.ui.Modal, title='🔗 貼上跳轉網址'):
    auth_url_input = discord.ui.TextInput(
        label='請貼上跳轉後的 localhost 完整網址',
        style=discord.TextStyle.long,
        placeholder='http://localhost/?state=...&code=4/0AeaY...',
        required=True,
        max_length=1000
    )

    def __init__(self, dept: str | None = None, grade: str | None = None, class_group: str = ""):
        super().__init__()
        self.dept = dept
        self.grade = grade
        self.class_group = class_group

    async def on_submit(self, interaction: discord.Interaction):
        await safe_defer(interaction)
        discord_id = str(interaction.user.id)
        username = interaction.user.display_name
        url = self.auth_url_input.value.strip()
        
        group_label = f" {self.class_group}班" if self.class_group else ""
        logger.info(f"📝 註冊請求 | 使用者：{username} (ID: {discord_id}) | 科系：{self.dept} ({DEPT_MAPPING.get(self.dept or '', self.dept)}) | 年級：{self.grade}{group_label}")
        
        try:
            verify_and_save_token(discord_id, url, self.dept, self.grade, self.class_group)
            dept_full = DEPT_MAPPING.get(self.dept or '', self.dept)
            logger.info(f"✅ 註冊成功 | {username} (ID: {discord_id}) | {self.dept} ({dept_full}) {self.grade}年級{group_label} | Token 已儲存")
            await safe_respond(
                interaction,
                f"✅ **註冊成功！**\n👤 身分綁定：**{self.dept} {self.grade}年級{group_label}**\n📅 Google 行事曆已連線！未來您可以直接問「我這學期的必修課有哪些？」",
            )
        except Exception as e:
            logger.error(f"❌ 註冊失敗 | {username} (ID: {discord_id}) | 錯誤：{e}")
            error_msg = str(e)
            hint = ""
            if "invalid_grant" in error_msg:
                if "code_verifier" in error_msg.lower() or "verifier" in error_msg.lower():
                    hint = "\n💡 **可能原因**：機器人在您登入 Google 後曾經重啟過，請重新執行 `/identity_login` 再操作一次。"
                else:
                    hint = "\n💡 **可能原因**：授權碼已過期（有效期約 10 分鐘），請重新執行 `/identity_login` 再操作一次。"
            elif "code" in error_msg.lower():
                hint = "\n💡 **可能原因**：網址被截斷了！請確認您複製了「完整的」網址（從 http 開頭到最後一個字元）。"
            await safe_respond(interaction, f"❌ 註冊失敗：{e}{hint}")


class CalendarAuthView(SafeView):
    def __init__(self, auth_url: str):
        super().__init__(timeout=900)  # 15 分鐘，匹配 Discord interaction token 上限
        self.selected_dept = None
        self.selected_grade = None
        self.add_item(discord.ui.Button(label="1. 點我登入 Google", url=auth_url, style=discord.ButtonStyle.link, row=0))

    @discord.ui.select(
        placeholder="2. 請選擇您的科系",
        options=[
            # ── 理工學院 ──
            discord.SelectOption(label="資訊工程學系", value="資工系", emoji="💻", description="理工學院"),
            discord.SelectOption(label="電機工程學系", value="電機系", emoji="⚡", description="理工學院"),
            discord.SelectOption(label="土木與工程管理學系", value="土木系", emoji="🏗️", description="理工學院"),
            discord.SelectOption(label="食品科學系", value="食品系", emoji="🧪", description="理工學院"),
            # ── 管理學院 ──
            discord.SelectOption(label="企業管理學系", value="企管系", emoji="📊", description="管理學院"),
            discord.SelectOption(label="觀光管理學系", value="觀光系", emoji="🏖️", description="管理學院"),
            discord.SelectOption(label="運動與休閒學系", value="運休系", emoji="🏃", description="管理學院"),
            discord.SelectOption(label="工業工程與管理學系", value="工管系", emoji="📐", description="管理學院"),
            # ── 人文社會學院 ──
            discord.SelectOption(label="國際暨大陸事務學系", value="國際系", emoji="🌏", description="人文社會學院"),
            discord.SelectOption(label="建築學系", value="建築系", emoji="🏛️", description="人文社會學院"),
            discord.SelectOption(label="海洋與邊境管理學系", value="海邊系", emoji="🌊", description="人文社會學院"),
            discord.SelectOption(label="應用英語學系", value="應英系", emoji="🇬🇧", description="人文社會學院"),
            discord.SelectOption(label="華語文學系", value="華語系", emoji="📝", description="人文社會學院"),
            discord.SelectOption(label="都市計畫與景觀學系", value="都景系", emoji="🌳", description="人文社會學院"),
            # ── 健康護理學院 ──
            discord.SelectOption(label="護理學系", value="護理系", emoji="⚕️", description="健康護理學院"),
            discord.SelectOption(label="長期照護學系", value="長照系", emoji="🏥", description="健康護理學院"),
            discord.SelectOption(label="社會工作學系", value="社工系", emoji="🤝", description="健康護理學院"),
            # ── 通識教育中心 ──
            discord.SelectOption(label="通識教育中心", value="通識中心", emoji="📚", description="通識教育中心"),
        ],
        row=1
    )
    async def select_dept(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.selected_dept = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="3. 請選擇您的年級/學制",
        options=[
            discord.SelectOption(label="🎓 大一", value="一", description="學士班一年級"),
            discord.SelectOption(label="🎓 大二", value="二", description="學士班二年級"),
            discord.SelectOption(label="🎓 大三", value="三", description="學士班三年級"),
            discord.SelectOption(label="🎓 大四", value="四", description="學士班四年級"),
            discord.SelectOption(label="🎓 大五", value="五", description="建築系五年制"),
            discord.SelectOption(label="📖 碩士班", value="碩", description="碩士班/碩士在職專班"),
            discord.SelectOption(label="🌙 進修部", value="進修", description="進修部/進修學士班"),
        ],
        row=2
    )
    async def select_grade(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.selected_grade = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="4. 甲乙班分組（選填，僅電機/企管等適用）",
        min_values=0,
        max_values=1,
        options=[
            discord.SelectOption(label="🅰️ 甲班", value="甲", description="適用於有甲乙分班的科系"),
            discord.SelectOption(label="🅱️ 乙班", value="乙", description="適用於有甲乙分班的科系"),
            discord.SelectOption(label="➖ 不適用", value="none", description="我的科系沒有分甲乙班"),
        ],
        row=3
    )
    async def select_class_group(self, interaction: discord.Interaction, select: discord.ui.Select):
        val = select.values[0] if select.values else "none"
        self.selected_class_group = "" if val == "none" else val
        await interaction.response.defer()

    @discord.ui.button(label="5. 點我完成綁定 (貼上網址)", style=discord.ButtonStyle.green, row=4, emoji="✅")
    async def enter_code_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.selected_dept or not self.selected_grade:
            await interaction.response.send_message("⚠️ 請先在上方下拉式選單選擇您的 **「科系」** 與 **「年級」** 喔！", ephemeral=True)
            return
        class_group = getattr(self, "selected_class_group", "")
        modal = CalendarAuthModal(dept=self.selected_dept, grade=self.selected_grade, class_group=class_group)
        await interaction.response.send_modal(modal)


# =========================================================================
# 🔑 身分管理 View（已註冊使用者）
# =========================================================================

class IdentityManageView(SafeView):
    """已註冊使用者的身分管理面板"""
    def __init__(self, discord_id: str, profile: dict):
        super().__init__(owner_id=discord_id)
        self.discord_id = discord_id
        self.profile = profile

    @discord.ui.button(label="🔄 重新註冊 (更改科系/年級)", style=discord.ButtonStyle.primary, row=0)
    async def re_register_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 此操作已處理過。")
            return
        await self.mark_handled(interaction)
        logger.info(f"🔄 重新註冊請求 | 使用者：{interaction.user.display_name} (ID: {self.discord_id})")
        delete_user_token(self.discord_id)
        auth_url = get_auth_url(self.discord_id)
        embed = discord.Embed(
            title="🔄 重新註冊 ─ 身分與 Google 行事曆綁定",
            description=(
                "────────────────────────\n"
                "✅ 您的舊資料已清除，請重新完成綁定：\n"
                "────────────────────────\n\n"
                "➀ 點擊下方 **`[點我登入 Google]`** 按鈕\n"
                "➁ 同意授權後，複製網址列**「整串網址」**\n"
                "➂ 選擇您的新**科系**與**年級**\n"
                "➃ 點擊綠色按鈕貼上網址\n"
            ),
            color=0x5865F2
        )
        embed.set_footer(text="💡 提示：請在 10 分鐘內完成操作，否則授權碼會過期")
        view = CalendarAuthView(auth_url)
        await safe_respond(interaction, embed=embed, view=view)

    @discord.ui.button(label="🗑️ 刪除帳號 (取消授權)", style=discord.ButtonStyle.danger, row=0)
    async def delete_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 此操作已處理過。")
            return
        confirm_view = DeleteConfirmView(self.discord_id)
        embed = discord.Embed(
            title="⚠️ 確認刪除帳號",
            description=(
                "────────────────────────\n"
                "此操作將會：\n\n"
                "❌ 移除您的科系、年級身分資料\n"
                "❌ 撤銷 Google 行事曆授權\n"
                "❌ 無法使用個人化課程推薦\n"
                "────────────────────────\n"
                "**⚠️ 此操作無法復原！**"
            ),
            color=discord.Color.red()
        )
        await safe_respond(interaction, embed=embed, view=confirm_view)

    @discord.ui.button(label="✖ 關閉", style=discord.ButtonStyle.secondary, row=0)
    async def cancel_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.mark_handled(interaction)
        await safe_respond(interaction, "👌 已關閉，您的資料沒有任何變動。")


class DeleteConfirmView(SafeView):
    """刪除確認面板"""
    def __init__(self, discord_id: str):
        super().__init__(owner_id=discord_id, timeout=120)
        self.discord_id = discord_id

    @discord.ui.button(label="✅ 確認刪除", style=discord.ButtonStyle.danger)
    async def confirm_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 此操作已處理過。")
            return
        await self.mark_handled(interaction)
        deleted = delete_user_token(self.discord_id)
        if deleted:
            logger.info(f"🗑️ 帳號已刪除 | 使用者：{interaction.user.display_name} (ID: {self.discord_id})")
            embed = discord.Embed(
                title="✅ 帳號已成功刪除",
                description="您的身分資料與 Google 授權已全部移除。\n\n💡 如需重新註冊，請再次使用 `/identity_login`",
                color=discord.Color.green()
            )
            await safe_respond(interaction, embed=embed)
        else:
            await safe_respond(interaction, "❓ 找不到您的註冊資料，可能已經被刪除了。")

    @discord.ui.button(label="✖ 取消", style=discord.ButtonStyle.secondary)
    async def cancel_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.mark_handled(interaction)
        await safe_respond(interaction, "👌 已取消刪除，您的資料安全無虞！")


# =========================================================================
# 🔑 主指令：/identity_login
# =========================================================================

@tree.command(name="identity_login", description="🔑 綁定您的 Google 帳號與身分，以啟用個人化檢索與行事曆功能")
async def slash_identity_login(interaction: discord.Interaction):
    discord_id = str(interaction.user.id)
    user_profile = get_user_profile(discord_id)
    
    if user_profile:
        dept = user_profile.get("department", "未知")
        grade = user_profile.get("grade", "未知")
        nickname = user_profile.get("nickname", "")
        nick_field = f"\n🏷️ **暱稱：** {nickname}" if nickname else ""
        
        embed = discord.Embed(
            title="👤 您的個人身份資訊",
            description=(
                f"────────────────────────\n"
                f"🏫 **科系：** {dept}\n"
                f"📚 **年級：** {grade}年級{nick_field}\n"
                f"📅 **Google 行事曆：** ✅ 已連線\n"
                f"────────────────────────\n\n"
                f"🔧 如需變更資料或取消授權，請使用下方按鈕。"
            ),
            color=0xFAA61A
        )
        embed.set_thumbnail(url=interaction.user.display_avatar.url)
        embed.set_footer(text=f"ID: {discord_id} · 注冊狀態：已認證 ✅")
        view = IdentityManageView(discord_id, user_profile)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        return
    
    await interaction.response.defer(ephemeral=True)
    auth_url = get_auth_url(discord_id)
    logger.info(f"🔑 新註冊流程啟動 | 使用者：{interaction.user.display_name} (ID: {discord_id})")
    
    embed = discord.Embed(
        title="🌟 歡迎來到 NQU 智慧校園助理註冊系統",
        description=(
            "────────────────────────\n"
            "請依照以下步驟完成身分綁定：\n"
            "────────────────────────\n\n"
            "➀ 點擊 **`[🔗 點我登入 Google]`** 按鈕\n"
            "➁ 同意授權後會跳轉到無效頁面 *(正常現象！)*\n"
            "➂ 複製頂部網址列的 **「整串網址」**\n"
            "➃ 選擇您的 **科系** 與 **年級**\n"
            "➄ 點擊綠色按鈕 **`[✅ 完成綁定]`** 貼上網址\n"
        ),
        color=0x57F287
    )
    embed.set_footer(text="💡 提示：請在 10 分鐘內完成操作，否則授權碼會過期")
    view = CalendarAuthView(auth_url)
    await interaction.followup.send(embed=embed, view=view, ephemeral=True)
