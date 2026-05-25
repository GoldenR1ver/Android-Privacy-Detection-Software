"""Configure matplotlib rcParams for Chinese text (Windows: register font file + sans-serif)."""

from __future__ import annotations

import os

_configured = False


def configure_matplotlib_chinese_font() -> None:
    """Call after matplotlib.use(...) and before heavy plotting."""
    global _configured
    if _configured:
        return

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False

    env_font = os.environ.get("MATPLOTLIB_ZH_FONT", "").strip()
    candidates = [
        p
        for p in (
            env_font,
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simfang.ttf",
        )
        if p
    ]

    for font_path in candidates:
        if not os.path.isfile(font_path):
            continue
        try:
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            name = prop.get_name()
            sans = plt.rcParams.get("font.sans-serif", [])
            if isinstance(sans, str):
                sans = [sans]
            plt.rcParams["font.sans-serif"] = [name] + [x for x in sans if x != name]
            plt.rcParams["font.family"] = "sans-serif"
            _configured = True
            return
        except Exception:
            continue

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["font.family"] = "sans-serif"
    _configured = True
