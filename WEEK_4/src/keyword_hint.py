from __future__ import annotations

import re
from typing import Pattern

# Heuristic stems for "personal information processing" discourse (Chinese).
_HINT_RE: Pattern[str] = re.compile(
    r"(个人信息|个人敏感信息|隐私|收集|处理|使用|存储|保存|共享|转让|"
    r"披露|委托|跨境|第三方|SDK|权限|设备信息|位置|通讯录|相册|相机|"
    r"麦克风|日志|Cookie|标识符|IMEI|OAID|IDFA|MAC|IP|账号|手机号|"
    r"实名|注销|撤回|删除|更正|复制|查阅|未成年人|儿童|敏感)"
)


def keyword_hint(text: str) -> bool:
    return bool(_HINT_RE.search(text))
