"""Configure matplotlib rcParams for Chinese text (Windows SimHei + fallbacks)."""

from __future__ import annotations

import os


def configure_matplotlib_chinese_font() -> None:
    """Call after matplotlib.use(...) and import matplotlib.pyplot as plt."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        if os.path.exists(font_path):
            custom_font = FontProperties(fname=font_path)
            plt.rcParams["font.family"] = custom_font.get_name()
        else:
            plt.rcParams["font.sans-serif"] = [
                "Arial Unicode MS",
                "SimHei",
                "Microsoft YaHei",
            ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        print("字体设置失败，可能需要手动安装中文字体")
