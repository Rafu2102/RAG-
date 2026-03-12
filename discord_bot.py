# -*- coding: utf-8 -*-
"""
discord_bot.py — Discord Bot 串接介面
======================================
負責監聽 Discord 訊息並將其傳入 RAG Pipeline。
包含獨立的對話記憶管理，確保不同頻道/使用者的對話不互相干擾。
"""

import os
import sys
import logging
import asyncio
from typing import Dict
import discord
from discord import app_commands
from dotenv import load_dotenv

import config
from rag.data_loader import load_and_index
from rag.query_router import init_known_registry
from llm.llm_answer import ConversationMemory
from main import rag_pipeline
from tools.calendar_tool import get_auth_url, verify_and_save_token, get_user_profile, delete_user_token

# 載入環境變數
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ADMIN_DISCORD_IDS = set(filter(None, os.getenv("ADMIN_DISCORD_IDS", "").split(",")))

if not DISCORD_BOT_TOKEN:
    print("❌ 錯誤：找不到 DISCORD_BOT_TOKEN。請確保目錄下有 .env 檔案並設定正確的 Token。")
    sys.exit(1)

# 初始化 logging
config.setup_logging()
logger = logging.getLogger("discord_bot")

# 準備 Discord Client
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

# ── 科系簡稱 ↔ 完整名稱對照表 ──
DEPT_MAPPING = {
    "資工系": "資訊工程學系", "電機系": "電機工程學系",
    "土木系": "土木與工程管理學系", "食品系": "食品科學系",
    "企管系": "企業管理學系", "觀光系": "觀光管理學系",
    "運休系": "運動與休閒學系", "工管系": "工業工程與管理學系",
    "國際系": "國際暨大陸事務學系", "建築系": "建築學系",
    "海邊系": "海洋與邊境管理學系", "應英系": "應用英語學系",
    "華語系": "華語文學系", "都景系": "都市計畫與景觀學系",
    "護理系": "護理學系", "長照系": "長期照護學系",
    "社工系": "社會工作學系", "通識中心": "通識教育中心",
}
# 反向映射（完整名稱 → 簡稱），方便反查
DEPT_REVERSE = {v: k for k, v in DEPT_MAPPING.items()}

# ── 全域狀態 ──
# 儲存不同頻道的對話記憶，Key 是 Channel ID
channel_memories: Dict[int, ConversationMemory] = {}
# 儲存全域向量索引
global_nodes = None
global_faiss = None
global_bm25 = None

# 【伺服器就緒門欄】確保機器人完全載入完畢後才接受指令
bot_ready = asyncio.Event()

# 【GPU 佇列排程】限制同時只有 1 個 RAG Pipeline 接觸 GPU，避免 VRAM 溢出崩潰
gpu_semaphore = asyncio.Semaphore(1)
_queue_waiters = 0  # 追蹤正在等待的使用者數量

def get_channel_memory(channel_id: int) -> ConversationMemory:
    """取得特定頻道的對話記憶，若無則建立"""
    if channel_id not in channel_memories:
        channel_memories[channel_id] = ConversationMemory()
    return channel_memories[channel_id]


def smart_split_message(text: str, max_len: int = 1900) -> list[str]:
    """將過長的訊息智慧分段，在段落邊界擷取，不破壞排版"""
    if len(text) <= max_len:
        return [text]
    
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        
        # 在 max_len 以內找最後一個換行分割點
        split_at = text.rfind('\n', 0, max_len)
        if split_at == -1 or split_at < max_len // 2:
            # 找不到換行就找最後一個空白
            split_at = text.rfind(' ', 0, max_len)
        if split_at == -1 or split_at < max_len // 2:
            split_at = max_len
        
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    
    # 加上續傳標記
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            chunks[i] += f"\n\n✅ *續讀下一則 ({i+2}/{len(chunks)})...*"
    
    return chunks


# =========================================================================
# 🌟 註冊原生斜線指令 (Slash Commands)
# =========================================================================

@tree.command(name="rebuild", description="🔄 重建並熱更新課程索引庫 (管理員專用)")
async def slash_rebuild(interaction: discord.Interaction):
    # 🔒 管理員權限檢查
    if ADMIN_DISCORD_IDS and str(interaction.user.id) not in ADMIN_DISCORD_IDS:
        await interaction.response.send_message("🚫 抱歉，只有管理員才能執行這個指令喔！", ephemeral=True)
        return
    
    if not bot_ready.is_set():
        await interaction.response.send_message("⏳ 機器人正在啟動中，請稍後再試！", ephemeral=True)
        return

    # 【防當機關鍵】呼叫 defer() 會先在聊天室顯示「機器人正在思考中...」
    # 這能幫我們向 Discord 爭取突破 3 秒限制的無限運算時間！
    await interaction.response.defer()
    
    try:
        # 在背景執行緒跑重建，避免卡死 Discord 機器人
        nodes, faiss_idx, bm25_idx = await asyncio.to_thread(load_and_index, True)
        global global_nodes, global_faiss, global_bm25
        global_nodes = nodes
        global_faiss = faiss_idx
        global_bm25 = bm25_idx
        init_known_registry(nodes)
        
        # 處理完成後，使用 followup.send 傳送最終結果
        await interaction.followup.send(f"✅ 索引重建與載入無縫完成！最新資料已上線（共 {len(nodes)} 個區段）。")
    except Exception as e:
        await interaction.followup.send(f"❌ 索引重建失敗：{e}")


@tree.command(name="clear", description="🗑️ 清除當前頻道的對話記憶")
async def slash_clear(interaction: discord.Interaction):
    if interaction.channel_id in channel_memories:
        channel_memories[interaction.channel_id].clear()
    await interaction.response.send_message("✅ 已經為您清除這個頻道的對話記憶囉！")


# =========================================================================
# 🌟 OAuth 授權與身分註冊介面
# =========================================================================

class CalendarAuthModal(discord.ui.Modal, title='🔗 貼上跳轉網址'):
    # 輸入框：授權網址（使用 long/paragraph 模式以容納完整 OAuth URL）
    auth_url_input = discord.ui.TextInput(
        label='請貼上跳轉後的 localhost 完整網址',
        style=discord.TextStyle.long,
        placeholder='http://localhost/?state=...&code=4/0AeaY...',
        required=True,
        max_length=1000
    )

    def __init__(self, dept: str, grade: str, class_group: str = ""):
        super().__init__()
        self.dept = dept
        self.grade = grade
        self.class_group = class_group

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        discord_id = str(interaction.user.id)
        username = interaction.user.display_name
        url = self.auth_url_input.value.strip()
        
        group_label = f" {self.class_group}班" if self.class_group else ""
        logger.info(f"📝 註冊請求 | 使用者：{username} (ID: {discord_id}) | 科系：{self.dept} ({DEPT_MAPPING.get(self.dept, self.dept)}) | 年級：{self.grade}{group_label}")
        
        try:
            # 呼叫升級版的儲存函式（傳入簡稱，calendar_tool 內部也會對應完整名稱）
            verify_and_save_token(discord_id, url, self.dept, self.grade, self.class_group)
            dept_full = DEPT_MAPPING.get(self.dept, self.dept)
            logger.info(f"✅ 註冊成功 | {username} (ID: {discord_id}) | {self.dept} ({dept_full}) {self.grade}年級{group_label} | Token 已儲存")
            await interaction.followup.send(
                f"✅ **註冊成功！**\n👤 身分綁定：**{self.dept} {self.grade}年級{group_label}**\n📅 Google 行事曆已連線！未來您可以直接問「我這學期的必修課有哪些？」", 
                ephemeral=True
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
            await interaction.followup.send(f"❌ 註冊失敗：{e}{hint}", ephemeral=True)

class CalendarAuthView(discord.ui.View):
    def __init__(self, auth_url: str):
        super().__init__(timeout=300)
        self.selected_dept = None
        self.selected_grade = None
        
        # 1. 登入按鈕
        self.add_item(discord.ui.Button(label="1. 點我登入 Google", url=auth_url, style=discord.ButtonStyle.link, row=0))

    # 2. 選擇科系（按學院分組，全校完整科系清單）
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

    # 3. 選擇年級/學制
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

    # 4. 選擇班級分組（選填 — 電機/企管等有甲乙班分組的科系使用）
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

    # 5. 註冊綁定 (呼叫 Modal)
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

class IdentityManageView(discord.ui.View):
    """已註冊使用者的身分管理面板：顯示資訊、重新註冊、刪除帳號"""
    def __init__(self, discord_id: str, profile: dict):
        super().__init__(timeout=120)
        self.discord_id = discord_id
        self.profile = profile

    @discord.ui.button(label="🔄 重新註冊 (更改科系/年級)", style=discord.ButtonStyle.primary, row=0)
    async def re_register_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"🔄 重新註冊請求 | 使用者：{interaction.user.display_name} (ID: {self.discord_id})")
        delete_user_token(self.discord_id)
        logger.info(f"🗑️ 舊資料已刪除 | 使用者：{interaction.user.display_name} (ID: {self.discord_id})")
        
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
            color=0x5865F2  # Discord Blurple
        )
        embed.set_footer(text="💡 提示：請在 10 分鐘內完成操作，否則授權碼會過期")
        view = CalendarAuthView(auth_url)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    @discord.ui.button(label="🗑️ 刪除帳號 (取消授權)", style=discord.ButtonStyle.danger, row=0)
    async def delete_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"⚠️ 刪除帳號請求 | 使用者：{interaction.user.display_name} (ID: {self.discord_id})")
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
        await interaction.response.send_message(embed=embed, view=confirm_view, ephemeral=True)

    @discord.ui.button(label="✖ 關閉", style=discord.ButtonStyle.secondary, row=0)
    async def cancel_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("👌 已關閉，您的資料沒有任何變動。", ephemeral=True)


class DeleteConfirmView(discord.ui.View):
    """刪除確認面板"""
    def __init__(self, discord_id: str):
        super().__init__(timeout=30)
        self.discord_id = discord_id

    @discord.ui.button(label="✅ 確認刪除", style=discord.ButtonStyle.danger)
    async def confirm_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        deleted = delete_user_token(self.discord_id)
        if deleted:
            logger.info(f"🗑️ 帳號已刪除 | 使用者：{interaction.user.display_name} (ID: {self.discord_id})")
            embed = discord.Embed(
                title="✅ 帳號已成功刪除",
                description=(
                    "您的身分資料與 Google 授權已全部移除。\n\n"
                    "💡 如需重新註冊，請再次使用 `/identity_login`"
                ),
                color=discord.Color.green()
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            await interaction.response.send_message("❓ 找不到您的註冊資料，可能已經被刪除了。", ephemeral=True)

    @discord.ui.button(label="✖ 取消", style=discord.ButtonStyle.secondary)
    async def cancel_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"↩️ 取消刪除 | 使用者：{interaction.user.display_name} (ID: {self.discord_id})")
        await interaction.response.send_message("👌 已取消刪除，您的資料安全無虞！", ephemeral=True)


# =========================================================================
# 🔑 身分登入/管理主指令
# =========================================================================

@tree.command(name="identity_login", description="🔑 綁定您的 Google 帳號與身分，以啟用個人化檢索與行事曆功能")
async def slash_identity_login(interaction: discord.Interaction):
    discord_id = str(interaction.user.id)
    user_profile = get_user_profile(discord_id)
    
    # ── 已註冊：顯示身分管理面板 ──
    if user_profile:
        dept = user_profile.get("department", "未知")
        grade = user_profile.get("grade", "未知")
        nickname = user_profile.get("nickname", "")
        nick_field = f"\n🏷️ **暱稱：** {nickname}" if nickname else ""
        
        logger.info(f"👤 已註冊使用者進入身分管理 | {interaction.user.display_name} (ID: {discord_id}) | {dept} {grade}年級")
        
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
            color=0xFAA61A  # Discord Gold
        )
        embed.set_thumbnail(url=interaction.user.display_avatar.url)
        embed.set_footer(text=f"ID: {discord_id} · 注冊狀態：已認證 ✅")
        view = IdentityManageView(discord_id, user_profile)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        return
    
    # ── 未註冊：走正常 OAuth 流程 ──
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
        color=0x57F287  # Discord Green
    )
    embed.set_footer(text="💡 提示：請在 10 分鐘內完成操作，否則授權碼會過期")
    view = CalendarAuthView(auth_url)
    await interaction.followup.send(embed=embed, view=view, ephemeral=True)


# =========================================================================
# 🌟 原生問答斜線指令 (附帶精美文字提示框)
# =========================================================================

@tree.command(name="ask", description="🤖 呼叫智慧校園助理為您解答課程問題")
@app_commands.describe(question="請輸入你想詢問的課程問題 (例如：星期二有什麼課？)")
async def slash_ask(interaction: discord.Interaction, question: str):
    
    # 1. 告訴 Discord 我們正在思考，爭取處理時間
    await interaction.response.defer()
    
    # 【就緒檢查】
    if not bot_ready.is_set():
        await interaction.followup.send("⏳ 機器人正在啟動中，請稍後再試！")
        return
    
    channel_id = interaction.channel_id
    memory = get_channel_memory(channel_id)

    try:
        discord_id = str(interaction.user.id)
        user_profile = get_user_profile(discord_id)

        # 【GPU 佇列】若前方有人在等待，通知使用者
        global _queue_waiters
        if gpu_semaphore.locked():
            _queue_waiters += 1
            logger.info(f"⏳ GPU 佇列 | {interaction.user.display_name} 正在等待（前方 {_queue_waiters} 人）")
            await interaction.followup.send(f"💬 前方還有 {_queue_waiters} 位同學在排隊，請稍等喔！")
        
        async with gpu_semaphore:
            _queue_waiters = max(0, _queue_waiters - 1)
            logger.info(f"🚀 GPU 取得 | {interaction.user.display_name} 開始處理問題：{question[:40]}")
            
            # 2. 呼叫我們強大的 RAG Pipeline
            answer = await asyncio.to_thread(
                rag_pipeline,
                question,
                global_nodes,
                global_faiss,
                global_bm25,
                memory,
                False,
                user_profile,
                discord_id
            )
        
        # 3. 回傳答案 (智慧分段)
        nick = user_profile.get("nickname", "") if user_profile else ""
        name_tag = f"{nick} " if nick else ""
        logger.info(f"✅ /ask 回答完成 | 使用者：{interaction.user.display_name} | 問題：{question[:50]} | 回答長度：{len(answer)} 字")
        final_reply = f"**👤 {name_tag}你問：** {question}\n\n**🤖 助理回答：**\n{answer}"
        parts = smart_split_message(final_reply)
        for part in parts:
            await interaction.followup.send(part)
        
    except Exception as e:
        logger.exception(f"❌ /ask 錯誤 | 使用者：{interaction.user.display_name} | 問題：{question[:50]}")
        await interaction.followup.send(f"❌ 抱歉，系統發生錯誤：{e}")


@tree.command(name="add_calendar", description="📅 專屬指令：將課程或學校事件快速加入 Google 行事曆")
@app_commands.describe(event="請輸入想加入的課程或事件 (例如：什麼時候停修申請 / 程式設計 / 演算法)")
async def slash_add_calendar(interaction: discord.Interaction, event: str):
    
    # 1. 告訴 Discord 我們正在思考
    await interaction.response.defer()
    
    # 【就緒檢查】
    if not bot_ready.is_set():
        await interaction.followup.send("⏳ 機器人正在啟動中，請稍後再試！")
        return
    
    channel_id = interaction.channel_id
    memory = get_channel_memory(channel_id)

    try:
        discord_id = str(interaction.user.id)
        user_profile = get_user_profile(discord_id)

        # 強制附加強烈意圖字詞，確保 Query Router 100% 走 calendar_action 路由
        augmented_query = f"{event} 幫我加到行事曆"
        
        # 【GPU 佇列】
        global _queue_waiters
        if gpu_semaphore.locked():
            _queue_waiters += 1
            logger.info(f"⏳ GPU 佇列 | {interaction.user.display_name} 正在等待（前方 {_queue_waiters} 人）")
            await interaction.followup.send(f"💬 前方還有 {_queue_waiters} 位同學在排隊，請稍等喔！")
        
        async with gpu_semaphore:
            _queue_waiters = max(0, _queue_waiters - 1)
            logger.info(f"🚀 GPU 取得 | {interaction.user.display_name} 開始處理行事曆：{event[:40]}")
            
            answer = await asyncio.to_thread(
                rag_pipeline,
                augmented_query,
                global_nodes,
                global_faiss,
                global_bm25,
                memory,
                False,
                user_profile,
                discord_id
            )
        
        # 智慧分段
        nick = user_profile.get("nickname", "") if user_profile else ""
        name_tag = f"{nick} " if nick else ""
        logger.info(f"✅ /add_calendar 回答完成 | 使用者：{interaction.user.display_name} | 事件：{event[:50]} | 回答長度：{len(answer)} 字")
        final_reply = f"**👤 {name_tag}你要求加入行事曆：** {event}\n\n**🤖 助理回報：**\n{answer}"
        parts = smart_split_message(final_reply)
        for part in parts:
            await interaction.followup.send(part)
        
    except Exception as e:
        logger.exception(f"❌ /add_calendar 錯誤 | 使用者：{interaction.user.display_name} | 事件：{event[:50]}")
        await interaction.followup.send(f"❌ 抱歉，行事曆新增發生錯誤：{e}")


# =========================================================================
# 🤖 Discord Events
# =========================================================================

@client.event
async def on_ready():
    global global_nodes, global_faiss, global_bm25
    logger.info(f"✅ 機器人 {client.user} 已成功登入 Discord！")
    
    # 【修正】直接呼叫 sync() 即可，它會自動把我們寫好的指令覆蓋到伺服器上
    try:
        synced = await tree.sync()
        logger.info(f"✅ 已成功同步 {len(synced)} 個斜線指令！")
    except Exception as e:
        logger.error(f"❌ 斜線指令同步失敗：{e}")
    
    # 啟動時自動載入向量資料庫與 Reranker 模型
    logger.info("📂 正在載入課程資料與 Reranker 索引...")
    try:
        nodes, faiss_idx, bm25_idx = load_and_index()
        global_nodes = nodes
        global_faiss = faiss_idx
        global_bm25 = bm25_idx
        
        # 動態建構教師 & 課程名冊（取代硬編碼清單）
        init_known_registry(nodes)
        
        # 預先載入 Reranker 模型，避免第一次查詢時卡頓超過 15 秒
        from rag.reranker import get_reranker
        get_reranker()
        
        logger.info(f"✅ 索引與模型載入完成！共 {len(nodes)} 個文件區段")
        await client.change_presence(activity=discord.Game(name="✅ 已就緒 · 輸入 / 查詢課程"))
        
        # 【關鍵】所有資源載入完畢，打開就緒門欄
        bot_ready.set()
        logger.info("🟢 伺服器就緒完畢，開始接受指令！")
    except Exception as e:
        logger.error(f"❌ 索引載入失敗：{e}")
        await client.close()

@client.event
async def on_message(message: discord.Message):
    # 忽略機器人自己發送的訊息
    if message.author == client.user:
        return

    # 處理開發者 !sync 指令（瞬間同步斜線指令到當前伺服器）
    if "!sync" in message.content.lower():
        # 🔒 管理員權限檢查
        if ADMIN_DISCORD_IDS and str(message.author.id) not in ADMIN_DISCORD_IDS:
            await message.reply("🚫 抱歉，只有管理員才能執行同步指令喔！")
            return
        if message.guild:
            try:
                tree.copy_global_to(guild=message.guild)
                synced = await tree.sync(guild=message.guild)
                await message.reply(f"✅ 開發者模式：已強制將 {len(synced)} 個斜線指令瞬間同步至此伺服器 ({message.guild.name})！\n\n💡 **重要提示**：同步完成後，請按 `Ctrl+R` (Windows) 或重新開啟手遊版 Discord APP，然後輸入 `/` (斜線) 才會看到最新的選單喔！")
                logger.info(f"✅ 手動 Guild Sync 完成 ({message.guild.name})： {len(synced)} 個指令")
            except Exception as e:
                await message.reply(f"❌ 同步失敗：{e}")
                logger.error(f"❌ 手動 Guild Sync 失敗：{e}")
        else:
            await message.reply("⚠️ 這個指令只能在伺服器頻道中使用喔！")
        return

    # 檢查是否是被 tag，或是以特定前綴開頭 (例如 !ask)
    is_mentioned = client.user in message.mentions
    is_prefixed = message.content.startswith("!ask")  # 把空格拿掉，增加容錯
    
    if not (is_mentioned or is_prefixed):
        return

    # 清除 tag 殘留字串或 !ask 前綴，取得真正的問題
    # 若被 tag 可能是 User ID 或 Role ID，使用正則安全清除
    import re
    content_clean = re.sub(r'<@&?\d+>', '', message.content)
    question = content_clean.strip()
    
    if question.startswith("!ask"):
        question = question.removeprefix("!ask").strip()
    
    if not question:
        await message.reply("請輸入你想要查詢的課程問題喔！")
        return

    # 系統指令全部交給 @tree.command 斜線指令處理，on_message 只負責課程問答

    channel_id = message.channel.id
    memory = get_channel_memory(channel_id)
    discord_id = str(message.author.id)
    user_profile = get_user_profile(discord_id)
    
    profile_tag = ""
    if user_profile:
        profile_tag = f" | 身分：{user_profile.get('department', '?')} {user_profile.get('grade', '?')}年級"
    logger.info(f"🔍 處理問題：{question[:60]} | 使用者：{message.author.display_name} (ID: {discord_id}){profile_tag}")

    # 【就緒檢查】
    if not bot_ready.is_set():
        await message.reply("⏳ 機器人正在啟動中，請稍後再試！")
        return

    async with message.channel.typing():
        try:
            # 【GPU 佇列】若前方有人在等待，通知使用者
            global _queue_waiters
            if gpu_semaphore.locked():
                _queue_waiters += 1
                logger.info(f"⏳ GPU 佇列 | {message.author.display_name} 正在等待（前方 {_queue_waiters} 人）")
                await message.reply(f"💬 前方還有 {_queue_waiters} 位同學在排隊，請稍等喔！")
            
            async with gpu_semaphore:
                _queue_waiters = max(0, _queue_waiters - 1)
                logger.info(f"🚀 GPU 取得 | {message.author.display_name} 開始處理問題：{question[:40]}")
                
                # 將耗時的同步 rag_pipeline 放到背景執行緒 (Background Thread) 中執行
                answer = await asyncio.to_thread(
                    rag_pipeline,
                    question,
                    global_nodes,
                    global_faiss,
                    global_bm25,
                    memory,
                    False,
                    user_profile,
                    discord_id
                )
            
            # 智慧分段回傳
            nick = user_profile.get("nickname", "") if user_profile else ""
            if nick:
                answer = f"**{nick}**，{answer}"
            logger.info(f"✅ @tag 回答完成 | 使用者：{message.author.display_name} | 問題：{question[:50]} | 回答長度：{len(answer)} 字")
            parts = smart_split_message(answer)
            for i, part in enumerate(parts):
                if i == 0:
                    await message.reply(part)
                else:
                    await message.channel.send(part)
            
        except Exception as e:
            logger.exception(f"❌ @tag 錯誤 | 使用者：{message.author.display_name} | 問題：{question[:50]}")
            error_msg = str(e)
            if "Connection refused" in error_msg or "Failed to establish a new connection" in error_msg:
                await message.reply("❌ 糟糕！我的 AI 核心 (Ollama) 似乎沒有啟動，請管理員檢查一下伺服器喔！")
            else:
                await message.reply(f"❌ 查詢時發生內部錯誤，請稍後再試：{e}")

if __name__ == "__main__":
    logger.info("啟動 Discord Bot...")
    client.run(DISCORD_BOT_TOKEN)
