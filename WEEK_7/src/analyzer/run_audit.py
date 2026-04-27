import argparse
import ast
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request


LAB3_LABELS = ["right_claim", "method_claim", "app_test_candidate"]
LAB3_LIST_FIELDS = ["right_types", "execution_channels"]
LAB3_TEXT_FIELDS = [
    "path_text",
    "target_data",
    "access_copy_type",
    "time_limit",
    "dynamic_test_goal",
    "usability_risk",
    "reason",
]
LAB3_ALLOWED_RIGHT_TYPES = {
    "inform_decision",
    "consent",
    "withdraw_consent",
    "access_copy",
    "correction",
    "deletion",
    "account_cancellation",
    "personalized_recommendation",
    "complaint",
    "other",
}
LAB3_ALLOWED_CHANNELS = {
    "app_ui",
    "web",
    "email",
    "phone",
    "customer_service",
    "mail",
    "offline",
    "unknown",
}
LAB3_ALLOWED_RISK = {"low", "medium", "high", "unknown"}
LAB3_ALLOWED_ACCESS_COPY_TYPE = {"copy", "non_copy", "unknown", ""}


SYSTEM_PROMPT = (
    "你是中文 Android 隐私政策与用户隐私权执行方式分析助手。"
    "你的依据是《个人信息保护法》中个人信息、个人信息处理、以及个人在处理活动中的权利。"
    "你的任务是从单条隐私政策句子中识别用户权利、执行渠道、可动态验证目标与可用性风险，"
    "用于 Lab3：APP 用户隐私权力执行方式的自动化识别与可用性评估。"
)


USER_PROMPT_TEMPLATE = """请分析“当前隐私政策句子”是否描述 APP 用户隐私权利及其执行方式。

法律与实验判断依据：
1. 个人信息：以电子或者其他方式记录的、与已识别或者可识别自然人有关的各种信息，不包括匿名化处理后的信息。典型 PII 包括姓名、手机号、邮箱、身份证号；其它个人信息包括位置、交易记录、浏览记录、搜索记录、设备信息等。
2. 个人信息处理：包括个人信息的收集、存储、使用、加工、传输、提供、公开、删除等。
3. 知情、决定权（第四十四条）：用户有权知道个人信息如何被处理，并有权限制或拒绝他人处理其个人信息。隐私政策若明确告知“收集哪些数据、为何收集、如何使用、如何存储/共享/删除”，属于知情权信号；若提供同意、拒绝、关闭、限制、选择等控制方式，属于决定权信号。
4. 访问权（第四十五条）：用户有权查阅、复制个人信息；重点看是否提供访问、查询、复制、下载副本的路径，以及可访问/复制的信息范围是否充分。访问权需要进一步区分“个人信息副本”和“非副本”：个人信息副本通常可下载保存到本地、可传输给第三方、常为机器可读格式，突出表述包括“下载个人信息副本”“导出数据”“导出个人信息”“下载数据副本”，目的在于实现数据可携带性；非副本通常是在 App 或网页界面内查看个人信息，嵌入在页面中，无法或不易导出为独立文件。
5. 修改权（第四十六条）：用户有权请求更正、补充不准确或不完整的个人信息；重点看是否提供更正、更改、修改、补充的对象范围和路径。
6. 删除权（第四十七条）：符合目的已实现/不再必要、停止服务、保存期限届满、撤回同意、违法违规处理等情形时，应主动删除或允许用户请求删除；注销账号也常是删除权的执行方式之一。

请围绕 Lab3 四个问题打标：
问题1：隐私政策声明与隐私法规是否一致？文本阶段主要看句子是否覆盖上述法定权利和个人信息处理对象；明显只写空泛法条、不说明数据或权利时风险更高。
问题2：是否提供用户权利执行方式？对应 method_claim。
问题3：执行方式可用性如何？对应 usability_risk，并依据路径清晰度、步骤复杂度、是否依赖客服/邮箱/人工审核判断。
问题4：执行方式有效性如何？单句文本无法最终证明有效性；若存在可测试路径，则 app_test_candidate=1，并在 dynamic_test_goal 中给出后续 Appium/网页/人工验证目标。

请输出严格 JSON，不要输出解释性正文。JSON 字段必须包含：
- "right_claim": 0 或 1。句子是否声明用户隐私权利、数据控制权或与个人信息处理相关的知情内容。
- "method_claim": 0 或 1。句子是否提供执行方式、渠道、路径、入口、联系方式、网页、开关或处理时限。
- "app_test_candidate": 0 或 1。句子是否适合进入 Appium/网页自动化或人工动态验证；只要出现 App 内路径、网页链接、设置入口、开关、下载/删除/注销流程、客服/邮箱等可验证执行方式，就应为 1。
- "right_types": 数组。可选值只从 ["inform_decision","consent","withdraw_consent","access_copy","correction","deletion","account_cancellation","personalized_recommendation","complaint","other"] 选择；无则 []。
- "execution_channels": 数组。可选值只从 ["app_ui","web","email","phone","customer_service","mail","offline","unknown"] 选择；无明确渠道但有执行方式时用 ["unknown"]。
- "path_text": 字符串。摘录最具体的执行路径、入口、链接、邮箱、电话或步骤；无则 ""。
- "target_data": 字符串。涉及的数据对象，例如“姓名、手机号、位置信息、浏览记录、个人信息副本、账号信息”；无则 ""。
- "access_copy_type": "copy"、"non_copy"、"unknown" 或 ""。仅在 right_types 包含 "access_copy" 时判断：能下载/导出/保存为本地文件或机器可读副本，填 "copy"；只能在 App/网页内查看，填 "non_copy"；访问权句子未说明是否可导出，填 "unknown"；非访问权句子填 ""。
- "time_limit": 字符串。处理时限或响应期限；无则 ""。
- "dynamic_test_goal": 字符串。建议后续验证目标，例如“在 App 设置页查找删除账号入口”；不适合动态验证则 ""。
- "usability_risk": "low"、"medium"、"high" 或 "unknown"。明确的一键入口/清晰 App 内路径为 low；需要多步但路径明确、网页跳转、邮箱/客服/人工审核为 medium；仅说“联系我们”“可申请”但无具体路径、范围很窄或条件不清为 high；无法判断为 unknown。
- "reason": 字符串。一句话说明判断依据，必须点明对应权利类型和路径/风险依据。

判标细则：
1. 只描述“我们收集/使用/存储/共享哪些个人信息及目的”，属于知情权，right_claim=1，right_types 包含 "inform_decision"；但若没有用户可执行路径，method_claim=0，app_test_candidate=0。
2. 出现“自行选择、拒绝、同意、授权、关闭、开启、限制处理、隐私设置检查、个性化推荐开关”等，属于决定权或同意/撤回同意相关信号。
3. “查询/查阅/访问/复制/下载/导出个人信息或副本”属于访问权；如果只允许查询少量账号资料，应在 reason 中说明范围有限、可用性风险较高。若出现“下载个人信息副本”“导出数据”“导出个人信息”“机器可读格式”“传输给第三方”等，access_copy_type="copy"；若只是“在账号资料页查看/查询/访问”且无导出文件能力，access_copy_type="non_copy"。
4. “更正/更改/修改/补充个人信息”属于修改权；如果仅限昵称、头像等少量资料，应说明范围有限。
5. “删除/清除个人信息、注销账号、撤回同意后删除”属于删除权；若只给客服或邮箱渠道，通常 usability_risk=medium；若路径含糊，risk=high。
6. “参考上下文”可能是同簇句子汇总，仅用于辅助理解主题；最终只给“当前隐私政策句子”打标和抽取字段。

Few-shot 示例：

句子：我们可能还会从可信的合作伙伴处收集您的相关信息，包括目录服务合作伙伴、营销合作伙伴以及安全保护合作伙伴提供的信息，以便防范滥用行为、提供广告和研究服务。
输出：{{"right_claim":1,"method_claim":0,"app_test_candidate":0,"right_types":["inform_decision"],"execution_channels":[],"path_text":"","target_data":"合作伙伴提供的相关信息、潜在客户信息、安全保护相关信息","access_copy_type":"","time_limit":"","dynamic_test_goal":"","usability_risk":"unknown","reason":"句子告知了个人信息来源和使用目的，属于知情权信号，但未提供用户执行方式。"}}

句子：我们会利用 Cookie、像素代码、本地存储、数据库和服务器日志收集和存储信息。
输出：{{"right_claim":1,"method_claim":0,"app_test_candidate":0,"right_types":["inform_decision"],"execution_channels":[],"path_text":"","target_data":"Cookie、像素代码、本地存储、数据库和服务器日志","access_copy_type":"","time_limit":"","dynamic_test_goal":"","usability_risk":"unknown","reason":"句子说明了信息收集和存储技术，属于个人信息处理的知情内容，但没有用户可执行路径。"}}

句子：您可以使用 Android 设备的“设置”应用开启或关闭设备的位置信息功能。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["inform_decision","consent"],"execution_channels":["app_ui"],"path_text":"Android 设备的设置应用开启或关闭设备的位置信息功能","target_data":"位置信息","access_copy_type":"","time_limit":"","dynamic_test_goal":"验证设备或 App 权限设置中是否可以开启或关闭位置信息权限","usability_risk":"low","reason":"句子给出了用户控制位置信息收集的明确设置入口，属于决定权/同意控制，路径清晰。"}}

句子：您可以在“隐私设置检查”部分查看和调整重要的隐私设置。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["inform_decision"],"execution_channels":["web"],"path_text":"隐私设置检查","target_data":"重要的隐私设置","access_copy_type":"","time_limit":"","dynamic_test_goal":"验证隐私设置检查页面是否存在并可调整隐私控制项","usability_risk":"low","reason":"句子提供了集中查看和调整隐私设置的入口，属于决定权执行方式，且路径明确。"}}

句子：点击底部“设置”——点击“订阅号消息个性推荐”，自行选择关闭订阅号消息中的推荐功能的个性化内容推荐服务。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["personalized_recommendation"],"execution_channels":["app_ui"],"path_text":"底部设置-订阅号消息个性推荐-关闭个性化内容推荐服务","target_data":"个性化内容推荐服务","access_copy_type":"","time_limit":"","dynamic_test_goal":"用 Appium 验证 App 内是否存在订阅号消息个性推荐关闭入口","usability_risk":"low","reason":"句子提供了关闭个性化推荐的具体 App 内路径，属于决定权执行方式且可动态验证。"}}

句子：拒绝提供该信息仅会使你无法使用上述功能，但不影响你正常使用微信的其他功能。
输出：{{"right_claim":1,"method_claim":0,"app_test_candidate":0,"right_types":["inform_decision","consent"],"execution_channels":[],"path_text":"","target_data":"上述功能所需信息","access_copy_type":"","time_limit":"","dynamic_test_goal":"","usability_risk":"unknown","reason":"句子说明用户可拒绝提供信息及拒绝后果，属于决定权/同意相关说明，但没有具体操作入口。"}}

句子：您可以在账号资料页查询您的昵称、头像、手机号等账号信息。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["access_copy"],"execution_channels":["app_ui"],"path_text":"账号资料页","target_data":"昵称、头像、手机号等账号信息","access_copy_type":"non_copy","time_limit":"","dynamic_test_goal":"验证账号资料页是否可查看上述个人信息，并确认是否仅为界面查看而无法导出副本","usability_risk":"medium","reason":"句子提供了访问个人信息的 App 内入口，但只是界面内查询账号资料，未体现可下载或导出个人信息副本，访问权覆盖可能有限。"}}

句子：您可以在数据导出页面下载个人信息副本，副本通常以机器可读格式提供，便于您保存或传输给第三方服务。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["access_copy"],"execution_channels":["web"],"path_text":"数据导出页面下载个人信息副本","target_data":"个人信息副本","access_copy_type":"copy","time_limit":"","dynamic_test_goal":"验证数据导出页面是否可下载机器可读的个人信息副本","usability_risk":"low","reason":"句子明确包含下载个人信息副本、机器可读格式和传输给第三方，属于可携带的数据副本。"}}

句子：您可以通过“设置-账号与安全-导出数据”申请导出您的账号资料、浏览记录和交易记录。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["access_copy"],"execution_channels":["app_ui"],"path_text":"设置-账号与安全-导出数据","target_data":"账号资料、浏览记录和交易记录","access_copy_type":"copy","time_limit":"","dynamic_test_goal":"用 Appium 验证 App 内是否存在导出数据入口，并记录可导出的文件类型和数据范围","usability_risk":"low","reason":"句子提供 App 内导出数据路径，导出意味着可保存为独立副本，属于访问权中的个人信息副本。"}}

句子：您有权查阅、复制您的个人信息。
输出：{{"right_claim":1,"method_claim":0,"app_test_candidate":0,"right_types":["access_copy"],"execution_channels":[],"path_text":"","target_data":"个人信息","access_copy_type":"unknown","time_limit":"","dynamic_test_goal":"","usability_risk":"high","reason":"句子声明访问权，但未说明查询、复制或下载副本的路径，也无法判断是否提供个人信息副本。"}}

句子：如您发现个人信息不准确或不完整，您可以在“我的-编辑资料”中更改头像、昵称和性别。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["correction"],"execution_channels":["app_ui"],"path_text":"我的-编辑资料","target_data":"头像、昵称和性别","access_copy_type":"","time_limit":"","dynamic_test_goal":"验证我的-编辑资料页面是否可以修改头像、昵称和性别","usability_risk":"medium","reason":"句子提供了修改权的 App 内路径，但可修改的信息范围较有限。"}}

句子：您可以通过“设置-账号与安全-注销账号”申请注销账号，注销后我们将删除或匿名化处理您的相关个人信息。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["deletion","account_cancellation"],"execution_channels":["app_ui"],"path_text":"设置-账号与安全-注销账号","target_data":"账号相关个人信息","access_copy_type":"","time_limit":"","dynamic_test_goal":"用 Appium 验证注销账号入口是否存在，并记录注销流程是否说明删除或匿名化结果","usability_risk":"low","reason":"句子给出了注销账号路径，并声明注销后删除或匿名化个人信息，属于删除权执行方式。"}}

句子：如需删除个人信息，请通过 privacy@example.com 联系我们，我们将在15个工作日内处理。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["deletion"],"execution_channels":["email"],"path_text":"privacy@example.com","target_data":"个人信息","access_copy_type":"","time_limit":"15个工作日","dynamic_test_goal":"验证邮箱删除请求渠道是否可用并记录响应要求","usability_risk":"medium","reason":"句子提供删除权邮箱渠道和处理时限，但依赖人工响应，可用性中等。"}}

句子：您可以联系我们请求删除您的个人信息。
输出：{{"right_claim":1,"method_claim":1,"app_test_candidate":1,"right_types":["deletion"],"execution_channels":["customer_service"],"path_text":"联系我们","target_data":"个人信息","access_copy_type":"","time_limit":"","dynamic_test_goal":"核查政策或 App 中是否存在具体联系方式，并验证删除请求入口是否可达","usability_risk":"high","reason":"句子声明删除权并给出笼统客服渠道，但没有具体路径、联系方式或时限，可用性风险高。"}}

参考上下文（可为空或为同簇句子汇总）：
{data_safety}

当前隐私政策句子：
{privacy_policy}
"""


def build_user_prompt(data_safety: str, privacy_policy: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        data_safety=data_safety,
        privacy_policy=privacy_policy,
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def count_data_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


class RuntimeLogger:
    def __init__(self, log_file: Optional[Path]) -> None:
        self._log_fp = None
        if log_file is not None:
            ensure_parent_dir(log_file)
            self._log_fp = log_file.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        if self._log_fp is not None:
            self._log_fp.write(line + "\n")
            self._log_fp.flush()

    def close(self) -> None:
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None


def _as_binary(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y"}:
            return 1
    return 0


def _as_str_list(value: object, allowed: set[str]) -> List[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    out: List[str] = []
    for item in raw_items:
        s = str(item).strip()
        if s in allowed and s not in out:
            out.append(s)
    return out


def normalize_prediction(d: Dict) -> Dict[str, object]:
    pred: Dict[str, object] = {label: _as_binary(d.get(label, 0)) for label in LAB3_LABELS}
    pred["right_types"] = _as_str_list(d.get("right_types"), LAB3_ALLOWED_RIGHT_TYPES)
    pred["execution_channels"] = _as_str_list(
        d.get("execution_channels"),
        LAB3_ALLOWED_CHANNELS,
    )
    for field in LAB3_TEXT_FIELDS:
        pred[field] = str(d.get(field, "") or "").strip()
    if pred["access_copy_type"] not in LAB3_ALLOWED_ACCESS_COPY_TYPE:
        pred["access_copy_type"] = "unknown" if "access_copy" in pred["right_types"] else ""
    if pred["usability_risk"] not in LAB3_ALLOWED_RISK:
        pred["usability_risk"] = "unknown"
    return pred


def extract_json_dict(text: str) -> Dict[str, object]:
    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    for candidate in [stripped]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return normalize_prediction(parsed)
        except Exception:
            pass

    match = re.search(r"\{[\s\S]*?\}", stripped)
    if match:
        fragment = match.group(0)
        try:
            parsed = json.loads(fragment)
            if isinstance(parsed, dict):
                return normalize_prediction(parsed)
        except Exception:
            try:
                parsed = ast.literal_eval(fragment)
                if isinstance(parsed, dict):
                    return normalize_prediction(parsed)
            except Exception:
                pass

    return normalize_prediction({})


class BaseProvider:
    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, object]:
        raise NotImplementedError


class MockProvider(BaseProvider):
    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, object]:
        text = privacy_policy
        right_patterns = {
            "inform_decision": (
                "收集",
                "使用",
                "存储",
                "保存",
                "共享",
                "提供",
                "公开",
                "个人信息",
                "位置信息",
                "浏览记录",
                "交易记录",
                "Cookie",
                "日志",
            ),
            "withdraw_consent": ("撤回", "取消授权", "关闭授权", "拒绝"),
            "access_copy": ("查阅", "复制", "访问", "下载", "副本", "导出", "查询", "查看"),
            "correction": ("更正", "补充", "修改"),
            "deletion": ("删除", "清除"),
            "account_cancellation": ("注销", "销户"),
            "personalized_recommendation": ("个性化", "定向推送", "推荐"),
            "complaint": ("投诉", "举报", "申诉"),
            "consent": ("同意", "授权"),
        }
        rights = [
            key
            for key, words in right_patterns.items()
            if any(word in text for word in words)
        ]
        channels: List[str] = []
        if any(word in text for word in ("设置", "页面", "页", "开关", "入口", "App", "APP", "应用内")):
            channels.append("app_ui")
        if any(word in text for word in ("http://", "https://", "网页", "网站", "链接")):
            channels.append("web")
        if any(word in text for word in ("邮箱", "邮件", "@")):
            channels.append("email")
        if any(word in text for word in ("电话", "热线")):
            channels.append("phone")
        if any(word in text for word in ("客服", "联系", "我们")):
            channels.append("customer_service")
        method = bool(channels) or any(word in text for word in ("路径", "方式", "申请", "提交", "发送"))
        if method and not channels:
            channels.append("unknown")
        app_test = method and bool(rights)
        risk = "unknown"
        if method:
            risk = "low" if "设置" in text or "开关" in text else "medium"
        if method and any(word in text for word in ("联系我们", "客服", "邮箱", "邮件", "申请")):
            risk = "medium"
        if rights and not method:
            risk = "high"
        access_copy_type = ""
        if "access_copy" in rights:
            if any(word in text for word in ("副本", "导出", "下载", "机器可读", "传输给第三方")):
                access_copy_type = "copy"
            elif any(word in text for word in ("查询", "查看", "访问", "账号资料", "页面")):
                access_copy_type = "non_copy"
            else:
                access_copy_type = "unknown"
        return normalize_prediction(
            {
                "right_claim": 1 if rights else 0,
                "method_claim": 1 if method else 0,
                "app_test_candidate": 1 if app_test else 0,
                "right_types": rights or ([] if not rights else ["other"]),
                "execution_channels": channels,
                "path_text": "",
                "target_data": "个人信息" if "个人信息" in text else "",
                "access_copy_type": access_copy_type,
                "time_limit": "",
                "dynamic_test_goal": "验证隐私权执行入口是否可用" if app_test else "",
                "usability_risk": risk,
                "reason": "mock：基于关键词规则的 Lab3 行权方式占位判断",
            }
        )


class DeepSeekProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("DeepSeek API key is empty.")
        try:
            self.api_key.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "DeepSeek API key contains non-ASCII characters. "
                "Please set a real API key (do not use placeholders like Chinese text)."
            ) from exc
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, object]:
        user_prompt = build_user_prompt(data_safety, privacy_policy)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "Failed to build HTTP headers. Check DEEPSEEK_API_KEY for invalid characters."
            ) from exc

        obj = json.loads(raw)
        content = (
            obj.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return extract_json_dict(content)


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        model: str = "qwen3.5:9b",
        base_url: str = "http://127.0.0.1:11434/api/chat",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, object]:
        user_prompt = build_user_prompt(data_safety, privacy_policy)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        content = obj.get("message", {}).get("content", "")
        return extract_json_dict(content)


class LocalHFProvider(BaseProvider):
    def __init__(
        self,
        model_id: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 256,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )

    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, object]:
        user_prompt = build_user_prompt(data_safety, privacy_policy)

        try:
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        out = self.generator(prompt)
        generated = out[0]["generated_text"] if out else "{}"
        return extract_json_dict(generated)


def make_provider(args: argparse.Namespace) -> BaseProvider:
    if args.provider == "mock":
        return MockProvider()
    if args.provider == "local":
        return LocalHFProvider(
            model_id=args.local_model_id,
            max_new_tokens=args.max_new_tokens,
        )
    if args.provider == "deepseek":
        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set and --api-key was not provided.")
        return DeepSeekProvider(
            api_key=api_key,
            model=args.deepseek_model,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    if args.provider == "ollama":
        return OllamaProvider(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            timeout=args.timeout,
        )
    raise ValueError(f"Unknown provider: {args.provider}")


def run_audit(
    input_csv: Path,
    output_csv: Path,
    provider: BaseProvider,
    limit: Optional[int],
    logger: RuntimeLogger,
    log_every: int,
    resume: bool = False,
    resume_from: Optional[int] = None,
) -> None:
    logger.log(
        f"[audit] start: input={input_csv}, output={output_csv}, limit={limit}, "
        f"resume={resume}, resume_from={resume_from}"
    )
    ensure_parent_dir(output_csv)
    if resume_from is not None and resume_from < 1:
        raise ValueError("--resume-from must be >= 1.")

    start_row = 1
    if resume_from is not None:
        start_row = resume_from
        logger.log(f"[audit] manual resume: start from row {start_row}")
    elif resume:
        completed = count_data_rows(output_csv)
        start_row = completed + 1
        logger.log(f"[audit] auto resume: detected {completed} completed rows, start from row {start_row}")

    write_mode = "a" if (start_row > 1 and output_csv.exists()) else "w"
    with input_csv.open("r", newline="", encoding="utf-8") as fin, output_csv.open(
        write_mode, newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        if "result" not in fieldnames:
            fieldnames.append("result")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()

        processed_count = 0
        total_seen = 0
        for row in reader:
            total_seen += 1
            if total_seen < start_row:
                continue
            if limit is not None and total_seen > limit:
                logger.log(f"[audit] reached limit={limit}, stop")
                break
            data_safety = row.get("data_safety_content", "")
            privacy_policy = row.get("privacy_policy_content", "")
            logger.log(f"[audit] row {total_seen}: request model inference")
            pred = provider.infer(data_safety, privacy_policy)
            logger.log(f"[audit] row {total_seen}: inference finished")
            row["result"] = json.dumps(pred, ensure_ascii=False)
            writer.writerow(row)
            processed_count += 1
            if processed_count % log_every == 0:
                logger.log(f"[audit] processed row {total_seen}")

    if total_seen < start_row:
        logger.log(
            f"[audit] warning: start_row={start_row} exceeds input rows={total_seen}, "
            "nothing processed"
        )
    logger.log(
        f"[audit] completed: processed_in_this_run={processed_count}, "
        f"last_input_row_seen={total_seen}"
    )


def postprocess_results(input_csv: Path, output_csv: Path, logger: RuntimeLogger) -> None:
    logger.log(f"[postprocess] start: input={input_csv}, output={output_csv}")
    ensure_parent_dir(output_csv)
    with input_csv.open("r", newline="", encoding="utf-8") as fin, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        if "result" in fieldnames:
            fieldnames.remove("result")
        for c in LAB3_LABELS + LAB3_LIST_FIELDS + LAB3_TEXT_FIELDS:
            if c not in fieldnames:
                fieldnames.append(c)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            pred = extract_json_dict(row.get("result", ""))
            row.pop("result", None)
            for k, v in pred.items():
                if isinstance(v, list):
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = str(v)
            writer.writerow(row)
            logger.log(f"[postprocess] processed row {idx}")
    logger.log("[postprocess] completed")


def evaluate_results(
    prediction_csv: Path,
    groundtruth_csv: Path,
    metrics_output_csv: Path,
    logger: RuntimeLogger,
    figures_dir: Optional[Path] = None,
) -> None:
    logger.log(
        f"[evaluate] start: prediction={prediction_csv}, groundtruth={groundtruth_csv}, "
        f"metrics={metrics_output_csv}, figures_dir={figures_dir}"
    )
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    plt = None
    if figures_dir is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        import matplotlib_zh

        matplotlib_zh.configure_matplotlib_chinese_font()

        plt = _plt

    labels = LAB3_LABELS
    pred_df = pd.read_csv(prediction_csv)
    gt_df = pd.read_csv(groundtruth_csv)

    for c in labels:
        pred_df[c] = pred_df[c].astype(int)
        gt_df[c] = gt_df[c].astype(int)

    rows = []
    for c in labels:
        y_true = gt_df[c]
        y_pred = pred_df[c]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        rows.append(
            {
                "label": c,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1]),
            }
        )
        logger.log(
            f"[evaluate] {c}: precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1:.4f}, accuracy={accuracy:.4f}"
        )

        if figures_dir is not None and plt is not None:
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(4.5, 4.0))
            plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix - {c}")
            plt.colorbar()
            plt.xticks([0, 1], ["0", "1"])
            plt.yticks([0, 1], ["0", "1"])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            fig_path = figures_dir / f"cm_{c}.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()

    ensure_parent_dir(metrics_output_csv)
    with metrics_output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.log(f"[evaluate] metrics written to: {metrics_output_csv}")
    logger.log("[evaluate] completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Android privacy audit pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    common_runtime = argparse.ArgumentParser(add_help=False)
    common_runtime.add_argument(
        "--log-file",
        default="results/run_audit_runtime.log",
        help="Path to runtime log file",
    )

    common_llm = argparse.ArgumentParser(add_help=False)
    common_llm.add_argument(
        "--provider",
        choices=["local", "deepseek", "ollama", "mock"],
        default="deepseek",
        help="LLM backend provider（默认 deepseek + deepseek-chat API）",
    )
    common_llm.add_argument("--local-model-id", default="microsoft/Phi-3-mini-4k-instruct")
    common_llm.add_argument("--max-new-tokens", type=int, default=512)
    common_llm.add_argument("--api-key", default=None, help="DeepSeek API key")
    common_llm.add_argument("--deepseek-model", default="deepseek-chat")
    common_llm.add_argument("--base-url", default="https://api.deepseek.com/chat/completions")
    common_llm.add_argument("--ollama-model", default="qwen3.5:9b")
    common_llm.add_argument("--ollama-base-url", default="http://127.0.0.1:11434/api/chat")
    common_llm.add_argument("--timeout", type=int, default=120)

    p_audit = sub.add_parser(
        "audit",
        parents=[common_llm, common_runtime],
        help="Run raw model audit",
    )
    p_audit.add_argument("--input-csv", required=True)
    p_audit.add_argument("--output-csv", required=True)
    p_audit.add_argument("--limit", type=int, default=None)
    p_audit.add_argument("--log-every", type=int, default=1)
    p_audit.add_argument("--resume", action="store_true", help="Resume from existing output CSV")
    p_audit.add_argument("--resume-from", type=int, default=None, help="1-based input row index to resume from")

    p_post = sub.add_parser("postprocess", parents=[common_runtime], help="Parse result column to labels")
    p_post.add_argument("--input-csv", required=True)
    p_post.add_argument("--output-csv", required=True)

    p_eval = sub.add_parser("evaluate", parents=[common_runtime], help="Evaluate predictions against groundtruth")
    p_eval.add_argument("--prediction-csv", required=True)
    p_eval.add_argument("--groundtruth-csv", required=True)
    p_eval.add_argument("--metrics-output-csv", required=True)
    p_eval.add_argument("--figures-dir", default=None)

    p_full = sub.add_parser("full", parents=[common_llm, common_runtime], help="Run full pipeline")
    p_full.add_argument("--input-csv", required=True)
    p_full.add_argument("--groundtruth-csv", required=True)
    p_full.add_argument("--raw-output-csv", required=True)
    p_full.add_argument("--processed-output-csv", required=True)
    p_full.add_argument("--metrics-output-csv", required=True)
    p_full.add_argument("--figures-dir", default=None)
    p_full.add_argument("--limit", type=int, default=None)
    p_full.add_argument("--log-every", type=int, default=1)
    p_full.add_argument("--resume", action="store_true", help="Resume from existing raw output CSV")
    p_full.add_argument("--resume-from", type=int, default=None, help="1-based input row index to resume from")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logger = RuntimeLogger(Path(args.log_file))

    try:
        logger.log(f"[main] command={args.command}")

        if args.command == "audit":
            provider = make_provider(args)
            run_audit(
                Path(args.input_csv),
                Path(args.output_csv),
                provider,
                args.limit,
                logger,
                max(1, int(args.log_every)),
                args.resume,
                args.resume_from,
            )
            return 0

        if args.command == "postprocess":
            postprocess_results(Path(args.input_csv), Path(args.output_csv), logger)
            return 0

        if args.command == "evaluate":
            evaluate_results(
                Path(args.prediction_csv),
                Path(args.groundtruth_csv),
                Path(args.metrics_output_csv),
                logger,
                Path(args.figures_dir) if args.figures_dir else None,
            )
            return 0

        if args.command == "full":
            provider = make_provider(args)
            run_audit(
                Path(args.input_csv),
                Path(args.raw_output_csv),
                provider,
                args.limit,
                logger,
                max(1, int(args.log_every)),
                args.resume,
                args.resume_from,
            )
            postprocess_results(Path(args.raw_output_csv), Path(args.processed_output_csv), logger)
            evaluate_results(
                Path(args.processed_output_csv),
                Path(args.groundtruth_csv),
                Path(args.metrics_output_csv),
                logger,
                Path(args.figures_dir) if args.figures_dir else None,
            )
            return 0
    finally:
        logger.log("[main] finished")
        logger.close()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
