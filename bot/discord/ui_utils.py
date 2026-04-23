# -*- coding: utf-8 -*-
"""
bot/ui_utils.py — Discord UI 安全基底元件
==========================================
解決 Discord 互動按鈕的常見問題：

1. ⏰ View timeout 太短 → 離開頁面回來按鈕就壞了
   → 統一設為 900 秒（15 分鐘，匹配 Discord interaction token 上限）

2. 🔁 重複點擊 → 重複送出請求
   → 按下按鈕後立即禁用所有按鈕 + 記錄已處理狀態

3. 💥 Interaction 過期 → NotFound / HTTPException
   → 所有互動回應都包裹在 safe_respond 中，優雅降級

4. 🔒 權限檢查 → 只有原始操作者能按按鈕
   → interaction_check 驗證使用者身份
"""

import logging
import discord  # type: ignore

logger = logging.getLogger("discord_bot")

# Discord interaction token 最長有效期 = 15 分鐘
MAX_VIEW_TIMEOUT = 900


class SafeView(discord.ui.View):
    """
    安全的 Discord View 基底類別。

    所有需要按鈕互動的 View 都應繼承此類別，自動獲得：
    - 15 分鐘超長 timeout
    - 重複點擊防護
    - 互動錯誤保護
    - 操作者身份驗證
    - 超時自動禁用按鈕
    """

    def __init__(self, *, owner_id: str | int | None = None, timeout: float = MAX_VIEW_TIMEOUT):
        """
        Args:
            owner_id: 允許操作此 View 的使用者 Discord ID（None = 不限制）
            timeout: View 存活時間（秒），預設 900 = 15 分鐘
        """
        super().__init__(timeout=timeout)  # type: ignore
        self.owner_id = str(owner_id) if owner_id else None
        self._handled = False  # 防止重複處理
        self._message: discord.Message | None = None  # 追蹤附著的訊息

    # ── 權限檢查：只有原始操作者能按 ──
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.owner_id and str(interaction.user.id) != self.owner_id:
            await safe_respond(interaction, "❌ 這不是你的操作面板喔！", ephemeral=True)
            return False
        return True

    # ── 超時處理：自動禁用所有按鈕 ──
    async def on_timeout(self):
        await self._disable_all_buttons()

    # ── 禁用所有按鈕（內部使用）──
    async def _disable_all_buttons(self):
        """禁用所有互動元件並更新訊息"""
        for item in self.children:
            if hasattr(item, "disabled"):
                item.disabled = True  # type: ignore

        if self._message:
            try:
                await self._message.edit(view=self)
            except (discord.NotFound, discord.HTTPException):
                pass  # 訊息已被刪除或 token 過期，靜默忽略

    # ── 標記為已處理 + 禁用按鈕（子類調用）──
    async def mark_handled(self, interaction: discord.Interaction):
        """
        標記此 View 已被處理。會立即禁用所有按鈕並更新訊息。
        子類在處理完業務邏輯後應該調用此方法。
        """
        self._handled = True
        for item in self.children:
            if hasattr(item, "disabled"):
                item.disabled = True  # type: ignore

        # 嘗試編輯原始訊息來更新按鈕狀態
        try:
            await interaction.message.edit(view=self)  # type: ignore
        except (discord.NotFound, discord.HTTPException, AttributeError):
            pass

        self.stop()

    @property
    def is_handled(self) -> bool:
        """檢查此 View 是否已經被處理過"""
        return self._handled


async def safe_respond(
    interaction: discord.Interaction,
    content: str | None = None,
    *,
    embed: discord.Embed | None = None,
    view: discord.ui.View | None = None,
    ephemeral: bool = True,
    **kwargs,
) -> bool:
    """
    安全地回應 Discord interaction，自動處理各種錯誤情境。

    嘗試順序：
    1. interaction.response.send_message（首次回應）
    2. interaction.followup.send（已 defer 或已回應過）
    3. 靜默失敗（interaction 已完全過期）

    Returns:
        True 如果成功回應，False 如果全部失敗
    """
    send_kwargs = {}
    if content:
        send_kwargs["content"] = content
    if embed:
        send_kwargs["embed"] = embed
    if view:
        send_kwargs["view"] = view
    send_kwargs["ephemeral"] = ephemeral
    send_kwargs.update(kwargs)

    # 嘗試 1: 直接回應
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(**send_kwargs)
            return True
    except (discord.NotFound, discord.HTTPException) as e:
        logger.debug(f"safe_respond: response.send_message 失敗: {e}")

    # 嘗試 2: 用 followup
    try:
        await interaction.followup.send(**send_kwargs)
        return True
    except (discord.NotFound, discord.HTTPException) as e:
        logger.debug(f"safe_respond: followup.send 失敗: {e}")

    # 全部失敗
    logger.warning(f"safe_respond: 無法回應 interaction (user={interaction.user.id})，interaction 可能已過期")
    return False


async def safe_defer(interaction: discord.Interaction, *, ephemeral: bool = True) -> bool:
    """
    安全地 defer interaction。

    Returns:
        True 如果成功 defer，False 如果失敗
    """
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=ephemeral)
            return True
    except (discord.NotFound, discord.HTTPException) as e:
        logger.debug(f"safe_defer 失敗: {e}")
    return False
