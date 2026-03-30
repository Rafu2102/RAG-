# tools/calendar_tool.py — 向後相容模組 (Compatibility Shim)
# ================================================
# 本檔案已拆分為以下三個模組：
#   - tools/auth.py          — 授權、Token 管理、使用者身分、廣播名單
#   - tools/calendar_api.py  — Google Calendar CRUD 操作
#   - tools/group_manager.py — 群組標籤與邀請碼管理
#
# 為了向後相容，此檔案重新匯出所有公開函式。
# 新程式碼請直接 import 上述三個模組。
# ================================================

# === 授權與身分 ===
from tools.auth import (  # noqa: F401
    get_user_token_path,
    get_auth_url,
    verify_and_save_token,
    get_user_profile,
    delete_user_token,
    get_service,
    get_targeted_users,
)

# === 行事曆 CRUD ===
from tools.calendar_api import (  # noqa: F401
    _parse_dt,
    _to_utc_rfc3339,
    _get_event_source_tag,
    find_duplicate_event,
    create_calendar_event,
    delete_calendar_events,
    list_calendar_events,
    update_calendar_event,
)

# === 群組管理 ===
from tools.group_manager import (  # noqa: F401
    add_user_group,
    remove_user_group,
    create_group,
    get_group_by_code,
    list_all_groups,
    get_group_info,
    delete_group,
)
