# -*- coding: utf-8 -*-
"""
bot/cmd_groups.py — 群組邀請系統
==================================
包含：
- GroupInviteView (接受/拒絕邀請按鈕)
- /join_group (學生透過邀請碼加入群組)
"""

import logging
import discord  # type: ignore
from discord import app_commands  # type: ignore

from bot import tree, logger
from bot.ui_utils import SafeView, safe_respond
from tools.group_manager import add_user_group, get_group_by_code


# =========================================================================
# 🏷️ 群組邀請 View（Accept / Decline）
# =========================================================================

class GroupInviteView(SafeView):
    """群組邀請卡按鈕：學生可選擇同意加入或婉拒。"""
    def __init__(self, group_name: str):
        super().__init__(timeout=None)  # Persistent view
        self.group_name = group_name

    @discord.ui.button(label="✅ 同意加入", style=discord.ButtonStyle.success, custom_id="accept_group_btn")
    async def accept_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 此邀請已處理過。")
            return
        await self.mark_handled(interaction)
        discord_id = str(interaction.user.id)
        success = add_user_group(discord_id, self.group_name)

        if success:
            embed = interaction.message.embeds[0].copy() if interaction.message.embeds else discord.Embed()
            embed.color = discord.Color.green()
            embed.title = f"🎉 成功加入群組：{self.group_name}"
            embed.description = "您已成功加入該小組！未來將會收到專屬的群組公告與通知。"
            try:
                await interaction.response.edit_message(embed=embed, view=self)
            except (discord.NotFound, discord.HTTPException):
                await safe_respond(interaction, f"🎉 成功加入群組「{self.group_name}」！")
            logger.info(f"🏷️ 群組邀請接受 | {interaction.user.display_name} 加入「{self.group_name}」")
        else:
            await safe_respond(
                interaction,
                "❌ 加入失敗：您似乎還沒註冊系統，請先使用 `/identity_login` 進行註冊！",
            )

    @discord.ui.button(label="❌ 婉拒", style=discord.ButtonStyle.danger, custom_id="decline_group_btn")
    async def decline_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.is_handled:
            await safe_respond(interaction, "⚠️ 此邀請已處理過。")
            return
        await self.mark_handled(interaction)
        embed = interaction.message.embeds[0].copy() if interaction.message.embeds else discord.Embed()
        embed.color = discord.Color.dark_gray()
        embed.title = f"🚫 已婉拒加入：{self.group_name}"
        embed.description = "您已婉拒邀請。如果改變心意，可以請管理員重新發送邀請。"
        try:
            await interaction.response.edit_message(embed=embed, view=self)
        except (discord.NotFound, discord.HTTPException):
            await safe_respond(interaction, f"🚫 已婉拒加入「{self.group_name}」")
        logger.info(f"🏷️ 群組邀請拒絕 | {interaction.user.display_name} 拒絕「{self.group_name}」")


# =========================================================================
# 🏷️ /join_group — 學生透過邀請碼加入群組
# =========================================================================

@tree.command(name="join_group", description="🏷️ 使用邀請碼加入群組 (由管理員提供)")
@app_commands.describe(code="管理員提供的 6 碼邀請碼")
async def slash_join_group(interaction: discord.Interaction, code: str):
    discord_id = str(interaction.user.id)
    
    group_name = get_group_by_code(code.strip().upper())
    if not group_name:
        await interaction.response.send_message(
            f"❌ 邀請碼 `{code}` 無效或已過期，請確認後再試。",
            ephemeral=True
        )
        return
    
    success = add_user_group(discord_id, group_name)
    if success:
        await interaction.response.send_message(
            f"🎉 成功加入群組「**{group_name}**」！\n未來您將收到該群組的專屬公告與通知。",
            ephemeral=True
        )
        logger.info(f"🏷️ 邀請碼加入 | {interaction.user.display_name} 加入「{group_name}」| 碼={code}")
    else:
        await interaction.response.send_message(
            f"❌ 加入失敗：您似乎還沒註冊系統，請先使用 `/identity_login` 進行註冊！",
            ephemeral=True
        )
