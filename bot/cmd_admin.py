# -*- coding: utf-8 -*-
"""
bot/cmd_admin.py — 管理員專用指令
===================================
包含：
- /rebuild (重建索引)
- /admin_broadcast (群體廣播)
- /admin_dm (單一私訊)
- /admin_invite (群組邀請)
- /admin_invite_code (產生邀請碼)
"""

import asyncio
import logging
import discord  # type: ignore
from discord import app_commands  # type: ignore

from bot import (
    tree, client, logger,
    ADMIN_DISCORD_IDS,
    bot_ready, global_nodes, global_faiss, global_bm25,
)
from rag.index_manager import load_and_index
from rag.query_router import init_known_registry
from tools.auth import get_targeted_users
from tools.group_manager import create_group, list_all_groups
from bot.cmd_groups import GroupInviteView
from bot.audit import send_audit_log
import bot as _bot  # 用於更新 global 狀態


# =========================================================================
# 🔄 /rebuild — 重建索引
# =========================================================================

@tree.command(name="rebuild", description="🔄 重建並熱更新課程索引庫 (管理員專用)")
async def slash_rebuild(interaction: discord.Interaction):
    if ADMIN_DISCORD_IDS and str(interaction.user.id) not in ADMIN_DISCORD_IDS:
        await interaction.response.send_message("🚫 抱歉，只有管理員才能執行這個指令喔！", ephemeral=True)
        return
    if not bot_ready.is_set():
        await interaction.response.send_message("⏳ 機器人正在啟動中，請稍後再試！", ephemeral=True)
        return

    await interaction.response.defer()
    try:
        nodes, faiss_idx, bm25_idx = await asyncio.to_thread(load_and_index, True)
        _bot.global_nodes = nodes
        _bot.global_faiss = faiss_idx
        _bot.global_bm25 = bm25_idx
        init_known_registry(nodes)
        await interaction.followup.send(f"✅ 索引重建與載入無縫完成！最新資料已上線（共 {len(nodes)} 個區段）。")
    except Exception as e:
        await interaction.followup.send(f"❌ 索引重建失敗：{e}")


# =========================================================================
# 🗑️ /clear — 清除對話記憶
# =========================================================================

@tree.command(name="clear", description="🗑️ 清除當前頻道的對話記憶")
async def slash_clear(interaction: discord.Interaction):
    from bot import channel_memories
    if interaction.channel_id in channel_memories:
        channel_memories[interaction.channel_id].clear()
    await interaction.response.send_message("✅ 已經為您清除這個頻道的對話記憶囉！")


# =========================================================================
# 🛡️ 共用工具：安全 defer（防 GPU 忙碌導致 interaction 超時）
# =========================================================================

async def _safe_defer(interaction: discord.Interaction, ephemeral: bool = True) -> bool:
    """
    嘗試 defer interaction。若 GPU 忙碌導致 event loop 延遲超過 3 秒，
    interaction 會過期 (NotFound)。此函式優雅處理，改用頻道訊息回報。
    Returns: True = defer 成功（可用 followup），False = 已過期（改用頻道訊息）。
    """
    try:
        await interaction.response.defer(ephemeral=ephemeral)
        return True
    except discord.NotFound:
        logger.warning(f"⚠️ interaction 已過期 (可能 GPU 繁忙)，將改用頻道訊息回報")
        return False


async def _send_reply(interaction: discord.Interaction, msg: str, deferred: bool):
    """根據 defer 是否成功，選擇用 followup 或頻道訊息回覆。"""
    if deferred:
        await interaction.followup.send(msg, ephemeral=True)
    elif interaction.channel:
        await interaction.channel.send(f"{interaction.user.mention} {msg}")


# =========================================================================
# 📢 廣播引擎
# =========================================================================

async def safe_broadcast_engine(
    admin_user: discord.User,
    target_ids: list[str],
    embed: discord.Embed,
    interaction_channel: discord.TextChannel = None,
    file_url: str = None,
    file_name: str = None,
):
    """背景安全發送引擎，具備速率保護、錯誤統計與終端機 Log。"""
    import aiohttp  # type: ignore
    success_count = 0
    forbidden_count = 0
    error_count = 0
    forbidden_users = []

    file_bytes = None
    if file_url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(file_url) as resp:
                    if resp.status == 200:
                        file_bytes = await resp.read()
        except Exception as e:
            logger.error(f"📢 附件下載失敗: {e}")

    logger.info(f"📢 廣播引擎啟動 | 管理員：{admin_user.display_name} | 目標：{len(target_ids)} 人 | 附件={'有' if file_bytes else '無'}")

    for i, uid in enumerate(target_ids, 1):
        try:
            user = await client.fetch_user(int(uid))
            if file_bytes:
                import io
                file_obj = discord.File(io.BytesIO(file_bytes), filename=file_name or "attachment")
                await user.send(embed=embed, file=file_obj)
            else:
                await user.send(embed=embed)
            success_count += 1
            logger.info(f"  📨 [{i}/{len(target_ids)}] ✅ 發送成功 → {user.display_name} ({uid})")
        except discord.Forbidden:
            forbidden_count += 1
            forbidden_users.append(uid)
        except discord.NotFound:
            error_count += 1
        except Exception as e:
            error_count += 1
            logger.error(f"  📨 [{i}/{len(target_ids)}] 🔴 發送失敗 → {uid}: {e}")
        await asyncio.sleep(0.5)

    logger.info(f"📢 廣播完成 | 成功={success_count} 拒絕={forbidden_count} 錯誤={error_count}")

    # 廣播完成 → 結果發到監控頻道
    audit_fields = [
        ("📊 統計", f"**目標**：{len(target_ids)}\n🟢 成功：{success_count}\n🟡 私訊關閉：{forbidden_count}\n🔴 錯誤：{error_count}", True),
        ("📝 公告標題", embed.title or "無標題", True),
    ]
    if forbidden_users:
        audit_fields.append((
            "🟡 私訊被拒絕",
            "\n".join(forbidden_users[:10]) + (f"\n...等 {len(forbidden_users)} 人" if len(forbidden_users) > 10 else ""),
            False
        ))
    await send_audit_log(
        title="📢 廣播作業完成報告",
        description=f"管理員 **{admin_user.display_name}** 執行的廣播已完成",
        color=discord.Color.green() if error_count == 0 else discord.Color.orange(),
        fields=audit_fields,
        admin_user=admin_user,
    )


# ── 群組名稱自動完成 ──
async def group_name_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    all_groups = list_all_groups()
    filtered = [g for g in all_groups if current.lower() in g.lower()][:25]
    return [app_commands.Choice(name=g, value=g) for g in filtered]


# =========================================================================
# 📢 /admin_broadcast — 支援附件 + 可選標題/內容 + GPU 防超時
# =========================================================================
#
# 使用方式：
#   /admin_broadcast attachment:檔案             ← 只發附件（標題自動填）
#   /admin_broadcast title:標題 content:內容      ← 只發文字
#   /admin_broadcast title:標題 content:內容 attachment:檔案  ← 文字+附件
#   以上都可搭配 dept / grade / group 篩選目標

@tree.command(name="admin_broadcast", description="📢 [管理員專用] 發送系統廣播公告給指定學生群體")
@app_commands.default_permissions(administrator=True)
@app_commands.describe(
    title="公告標題 (可選，不填則自動產生)",
    content="公告內容 (可選，不填則留空)",
    attachment="附件檔案 (可選，直接上傳)",
    dept="目標科系 (不填=全校)", grade="目標年級 (不填=全系)",
    group="目標群組標籤 (不填=不限群組)"
)
@app_commands.autocomplete(group=group_name_autocomplete)
async def slash_admin_broadcast(
    interaction: discord.Interaction,
    title: str = None, content: str = None,
    attachment: discord.Attachment = None,
    dept: str = None, grade: str = None, group: str = None,
):
    if not ADMIN_DISCORD_IDS or str(interaction.user.id) not in ADMIN_DISCORD_IDS:
        try:
            await interaction.response.send_message("🚫 抱歉，這是管理員專用指令喔！", ephemeral=True)
        except discord.NotFound:
            pass
        return

    # 至少要有 title / content / attachment 其中一個
    if not title and not content and not attachment:
        try:
            await interaction.response.send_message(
                "❌ 請至少提供 **標題**、**內容** 或 **附件** 其中一項！", ephemeral=True
            )
        except discord.NotFound:
            pass
        return

    # ⚡ 嘗試 defer（若 GPU 忙碌可能失敗）
    deferred = await _safe_defer(interaction, ephemeral=True)

    # 篩選目標使用者
    targets = get_targeted_users(dept, grade, group)
    target_ids = [t["discord_id"] for t in targets]

    if not target_ids:
        parts = []
        if dept: parts.append(dept)
        if grade: parts.append(f"{grade}年級")
        if group: parts.append(f"標籤:[{group}]")
        filter_desc = " ".join(parts) if parts else "全校"
        await _send_reply(interaction, f"❌ 在 **{filter_desc}** 中找不到任何已註冊的使用者。", deferred)
        return

    filter_parts = []
    if dept: filter_parts.append(dept)
    if grade: filter_parts.append(f"{grade}年級")
    if group: filter_parts.append(f"標籤:[{group}]")
    filter_msg = " ".join(filter_parts) if filter_parts else "全校"

    # 自動填充標題
    final_title = title or ("📎 資料分享" if attachment else "📢 系統公告")
    final_content = content or ""
    file_url = attachment.url if attachment else None
    file_name = attachment.filename if attachment else None

    confirm_msg = (
        f"🚀 廣播任務已啟動！\n"
        f"📋 目標對象：**{filter_msg}** (共 {len(target_ids)} 人)\n"
        f"📝 標題：**{final_title}**"
        + (f"\n📎 附件：{file_name}" if file_name else "")
        + "\n\n稍後機器人將會以私訊回報執行結果給您。"
    )
    await _send_reply(interaction, confirm_msg, deferred)

    # 構建 Embed
    embed = discord.Embed(
        title=f"📢 {final_title}",
        description=final_content if final_content else None,
        color=discord.Color.gold(),
        timestamp=discord.utils.utcnow()
    )
    embed.set_footer(text=f"來自 NQU 校園智慧助理 | 發送者: {interaction.user.display_name}")
    embed.set_thumbnail(url=interaction.user.display_avatar.url)

    client.loop.create_task(
        safe_broadcast_engine(interaction.user, target_ids, embed, interaction.channel, file_url, file_name)
    )


# =========================================================================
# 💬 /admin_dm — 支援附件 + 可選內容 + GPU 防超時
# =========================================================================

@tree.command(name="admin_dm", description="💬 [管理員專用] 透過機器人私訊特定使用者")
@app_commands.default_permissions(administrator=True)
@app_commands.describe(
    target_user="要傳訊的對象",
    message="訊息內容 (可選，不填則只發附件)",
    attachment="附件檔案 (可選，直接上傳)"
)
async def slash_admin_dm(
    interaction: discord.Interaction,
    target_user: discord.User,
    message: str = None,
    attachment: discord.Attachment = None
):
    if not ADMIN_DISCORD_IDS or str(interaction.user.id) not in ADMIN_DISCORD_IDS:
        try:
            await interaction.response.send_message("🚫 抱歉，這是管理員專用指令喔！", ephemeral=True)
        except discord.NotFound:
            pass
        return

    if not message and not attachment:
        try:
            await interaction.response.send_message(
                "❌ 請至少提供 **訊息內容** 或 **附件** 其中一項！", ephemeral=True
            )
        except discord.NotFound:
            pass
        return

    deferred = await _safe_defer(interaction, ephemeral=True)

    try:
        embed = discord.Embed(
            title="💬 來自管理員的訊息",
            description=message if message else None,
            color=discord.Color.blue(), timestamp=discord.utils.utcnow()
        )
        embed.set_footer(text=f"來自 NQU 校園智慧助理 | 發送者: {interaction.user.display_name}")
        
        if attachment:
            import aiohttp, io  # type: ignore
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status == 200:
                            file_bytes = await resp.read()
                            file_obj = discord.File(io.BytesIO(file_bytes), filename=attachment.filename)
                            await target_user.send(embed=embed, file=file_obj)
                        else:
                            await target_user.send(embed=embed)
            except Exception:
                await target_user.send(embed=embed)
        else:
            await target_user.send(embed=embed)
        
        await _send_reply(interaction, f"✅ 已成功將訊息發送給 **{target_user.display_name}**！", deferred)
        # DM 審計 Log
        await send_audit_log(
            title="💬 管理員私訊",
            description=f"管理員 **{interaction.user.display_name}** 私訊了 **{target_user.display_name}**",
            fields=[("📝 內容", message[:200] if message else "僅附件", False)],
            admin_user=interaction.user,
        )
    except discord.Forbidden:
        await _send_reply(interaction, f"❌ 發送失敗：**{target_user.display_name}** 關閉了伺服器私訊功能。", deferred)
    except Exception as e:
        await _send_reply(interaction, f"❌ 發生錯誤：{e}", deferred)


# =========================================================================
# 🏷️ /admin_invite (批量邀請)
# =========================================================================

@tree.command(name="admin_invite", description="🏷️ [管理員專用] 邀請學生加入特定標籤群組 (支援一次最多5人)")
@app_commands.default_permissions(administrator=True)
@app_commands.describe(
    group_name="群組名稱 (可快選或輸入新群組)",
    user1="要邀請的學生 1", user2="要邀請的學生 2 (可選)",
    user3="要邀請的學生 3 (可選)", user4="要邀請的學生 4 (可選)",
    user5="要邀請的學生 5 (可選)",
)
@app_commands.autocomplete(group_name=group_name_autocomplete)
async def slash_admin_invite(
    interaction: discord.Interaction, group_name: str, user1: discord.User,
    user2: discord.User = None, user3: discord.User = None,
    user4: discord.User = None, user5: discord.User = None,
):
    if not ADMIN_DISCORD_IDS or str(interaction.user.id) not in ADMIN_DISCORD_IDS:
        try:
            await interaction.response.send_message("🚫 抱歉，這是管理員專用指令喔！", ephemeral=True)
        except discord.NotFound:
            pass
        return

    deferred = await _safe_defer(interaction, ephemeral=True)
    invite_code = create_group(group_name, str(interaction.user.id), interaction.user.display_name)

    target_users = [u for u in [user1, user2, user3, user4, user5] if u is not None]
    success_list, fail_list = [], []

    for target_user in target_users:
        try:
            embed = discord.Embed(
                title="💌 系統群組邀請",
                description=(
                    f"您好！管理員 **{interaction.user.display_name}** 邀請您加入：\n\n"
                    f"### 🏷️ 「{group_name}」\n\n"
                    f"加入後，您將能接收到該群組的專屬重要通知。\n請問您是否同意加入？"
                ),
                color=discord.Color.purple(), timestamp=discord.utils.utcnow()
            )
            embed.set_footer(text="NQU 校園智慧助理 · 群組管理系統")
            embed.set_thumbnail(url=interaction.user.display_avatar.url)
            view = GroupInviteView(group_name)
            await target_user.send(embed=embed, view=view)
            success_list.append(target_user.display_name)
        except discord.Forbidden:
            fail_list.append(f"{target_user.display_name} (私訊關閉)")
        except Exception as e:
            fail_list.append(f"{target_user.display_name} ({e})")
        await asyncio.sleep(0.3)

    result_msg = f"📋 邀請結果：群組「**{group_name}**」| 邀請碼：`{invite_code}`\n"
    if success_list:
        result_msg += f"\n✅ 成功發送 ({len(success_list)} 人)：{', '.join(success_list)}"
    if fail_list:
        result_msg += f"\n❌ 發送失敗 ({len(fail_list)} 人)：{', '.join(fail_list)}"
    result_msg += f"\n\n🔗 分享邀請碼給學生，他們可使用 `/join_group code:{invite_code}` 加入！"
    await _send_reply(interaction, result_msg, deferred)


# =========================================================================
# 🔗 /admin_invite_code
# =========================================================================

@tree.command(name="admin_invite_code", description="🔗 [管理員專用] 產生群組邀請碼")
@app_commands.default_permissions(administrator=True)
@app_commands.describe(group_name="群組名稱 (可快選或輸入新群組)")
@app_commands.autocomplete(group_name=group_name_autocomplete)
async def slash_admin_invite_code(interaction: discord.Interaction, group_name: str):
    if not ADMIN_DISCORD_IDS or str(interaction.user.id) not in ADMIN_DISCORD_IDS:
        try:
            await interaction.response.send_message("🚫 抱歉，這是管理員專用指令喔！", ephemeral=True)
        except discord.NotFound:
            pass
        return

    invite_code = create_group(group_name, str(interaction.user.id), interaction.user.display_name)
    share_msg = (
        f"🎉 **你已被邀請加入「{group_name}」！**\n\n"
        f"請在 Discord 輸入以下指令加入：\n```\n/join_group code:{invite_code}\n```\n"
        f"加入後將收到專屬公告與通知！"
    )
    try:
        await interaction.response.send_message(
            f"✅ 群組「**{group_name}**」的邀請碼：\n\n🔑 邀請碼： **`{invite_code}`**\n\n"
            f"學生使用 `/join_group code:{invite_code}` 即可加入\n\n--- 以下是可複製分享的訊息 ---\n{share_msg}",
            ephemeral=True
        )
    except discord.NotFound:
        if interaction.channel:
            await interaction.channel.send(
                f"{interaction.user.mention} ✅ 群組「**{group_name}**」的邀請碼：**`{invite_code}`**"
            )
    logger.info(f"🔗 邀請碼產生 | {interaction.user.display_name} | 群組={group_name} 邀請碼={invite_code}")
