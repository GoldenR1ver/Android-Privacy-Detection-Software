"""Configuration for Part 2 dynamic testing with Appium + MuMu emulator."""

from __future__ import annotations

import os

# ── Appium / MuMu emulator settings ──────────────────────────────────────────

# MuMu 模拟器默认 ADB 连接端口
# MuMu 6: 127.0.0.1:7555
# MuMu 12: 127.0.0.1:16384 (或 5555)
# MuMu X: 127.0.0.1:16384
MUMU_HOST = "127.0.0.1"
MUMU_ADB_PORT = 16384

APPIUM_HOST = "127.0.0.1"
APPIUM_PORT = 4723

# Appium desired capabilities
APPIUM_CAPS = {
    "platformName": "Android",
    "appium:automationName": "UiAutomator2",
    "appium:noReset": True,
    "appium:autoGrantPermissions": True,
    "appium:newCommandTimeout": 30,
    "appium:ensureWebviewsHavePages": True,
    "appium:nativeWebScreenshot": True,
}

IMPLICIT_WAIT_SEC = 1
ELEMENT_WAIT_SEC = 6
COMMAND_TIMEOUT_SEC = 30

# ── Target apps (top 12 by candidate count) ──────────────────────────────────

TARGET_APPS = [
    "51Job",
    "平安证券",
    "智联招聘",
    "BOSS直聘",
    "360借条",
    "58同城",
    "京东健康",
    "Soul",
    "京东金融",
    "Keep",
    "支付宝",
    "中国工商银行",
]

# App name → Android package name
# These need to be manually confirmed per APK
APP_PKG_MAP = {
    "51Job": "com.mvwcorp.job",
    "平安证券": "com.hundsun.pingan.android",
    "智联招聘": "com.zhaopin.social",
    "BOSS直聘": "com.hpbr.bosszhipin",
    "360借条": "com.qihoo.loan",
    "58同城": "com.wuba",
    "京东健康": "com.jd.health",
    "Soul": "cn.soulapp.android",
    "京东金融": "com.jd.jrapp",
    "Keep": "com.gotokeep.keep",
    "支付宝": "com.eg.android.AlipayGphone",
    "中国工商银行": "com.icbc",
    "小红书": "com.xingin.xhs",
}

# ── Right type classification ────────────────────────────────────────────────

CONSENT_RIGHTS = ["consent", "withdraw_consent"]
ACCESS_RIGHTS = ["access_copy"]
DELETION_RIGHTS = ["deletion", "account_cancellation"]

ALL_RIGHT_TYPES = CONSENT_RIGHTS + ACCESS_RIGHTS + DELETION_RIGHTS

# ── Scoring weights ──────────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    "findability": 0.30,
    "functionality": 0.35,
    "efficiency": 0.20,
    "consistency": 0.15,
}

# ── APK auto-install ──────────────────────────────────────────────────────

# 存放 APK 文件的目录，文件名需与 APP 名称一致（如 Keep.apk、支付宝.apk）
# 目录不存在或为空时跳过自动安装
APK_DIR = None  # 设为 Path 对象可启用，如 Path("data/apk")

# ── DeepSeek API ─────────────────────────────────────────────────────────

# 从环境变量读取，禁止把真实 key 写进仓库。
# PowerShell: $env:DEEPSEEK_API_KEY="your-key"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"

# ── Limits ───────────────────────────────────────────────────────────────────

DEFAULT_LIMIT_PER_APP_PER_RIGHT = 10
MAX_TEST_CASES_TOTAL = 500

# ── Channel grouping ────────────────────────────────────────────────────────

AUTOMATED_CHANNELS = {"app_ui"}
NON_AUTOMATED_CHANNELS = {"web", "email", "phone", "customer_service", "mail", "offline", "unknown"}

# ── Navigation aliases and locator hints ────────────────────────────────────

# Canonical navigation labels -> common synonyms across apps/locales
UI_TEXT_ALIASES = {
    "我": ["我", "我的", "个人", "个人中心", "Me", "Profile", "Account"],
    "菜单": ["菜单", "侧边栏", "更多", "更多选项", "导航菜单", "Hamburger"],
    "设置": ["设置", "系统设置", "偏好设置", "Settings", "Setting"],
    "账号与安全": ["账号与安全", "账户与安全", "账号安全", "账户安全", "Security"],
    "隐私": ["隐私", "隐私设置", "隐私中心", "Privacy"],
    "用户保护中心": ["用户保护中心", "安全中心", "账号安全中心", "User Protection"],
}

# Optional app-specific locator hints. Keys are app display names.
# Each entry maps canonical label -> locator candidates.
# Fill in resource ids for your target app versions over time.
APP_LOCATOR_HINTS = {
    "支付宝": {
        "我": {
            "ids": [],
            "descs": ["我的", "我", "Me", "个人中心"],
            "texts": ["我的", "我"],
            "bottom_index": 4,
        },
        "设置": {
            "ids": [],
            "descs": ["设置", "Settings"],
            "texts": ["设置"],
        },
        "用户保护中心": {
            "ids": [],
            "descs": ["用户保护中心", "安全中心"],
            "texts": ["用户保护中心", "安全中心"],
        },
    },
    "Keep": {
        "我": {
            "ids": [],
            "descs": ["我", "我的", "Me", "个人中心"],
            "texts": ["我", "我的"],
            "bottom_index": 4,
        },
        "菜单": {
            "ids": ["com.gotokeep.keep:id/viewToolbarSidebar"],
            "descs": ["打开侧边栏", "侧边栏", "导航菜单", "菜单", "更多", "更多选项"],
            "texts": [],
            "top_left_icon": True,
            "open_check_ids": ["com.gotokeep.keep:id/view_main_slide"],
            "open_check_texts": ["设置", "帮助中心"],
        },
        "设置": {
            "ids": [],
            "descs": ["设置", "Settings", "更多", "菜单"],
            "texts": ["设置"],
            "xpaths": ['//*[@resource-id="com.gotokeep.keep:id/textName" and @text="设置"]'],
            "requires_menu_open": True,
        },
    },
    "小红书": {
        "我": {
            "ids": [],
            "descs": ["我", "我的", "Me", "个人主页"],
            "texts": ["我", "我的"],
            "bottom_index": 4,
        },
        "菜单": {
            "ids": [],
            "descs": ["菜单", "侧边栏", "导航菜单", "更多", "更多选项"],
            "texts": ["菜单"],
            "top_left_icon": True,
            "open_check_texts": ["帮助与客服", "扫一扫", "设置"],
            "xpaths": [
                '//*[@content-desc="菜单"]',
            ],
        },
        "设置": {
            "ids": [],
            "descs": ["设置", "Settings", "偏好设置"],
            "texts": ["设置"],
            "xpaths": [
                '//*[@content-desc="设置"]',
                '//*[@text="设置"]',
            ],
            "requires_menu_open": True,
        },
    },
}

# App-specific parallel navigation groups.
# If path_text contains sequential peers inside the same group, executor will
# treat them as OR branches (tap-any), not strict serial taps.
APP_PARALLEL_PATH_GROUPS = {
    "小红书": [
        {"anchor": "我", "options": ["笔记", "收藏", "赞过"]},
    ],
}

# Access-copy capability baseline and fallback probing strategy.
# export_copy_in_app=False means "导出副本" is likely a product capability gap.
ACCESS_COPY_BASELINES = {
    "小红书": {
        "export_copy_in_app": False,
        "alt_channel_keywords": [
            "帮助与客服", "客服", "在线客服", "意见反馈", "联系我们", "邮箱",
        ],
        "alt_channel_paths": [
            "设置 - 帮助与客服",
            "我 - 设置 - 帮助与客服",
            "帮助与客服",
        ],
    },
}
