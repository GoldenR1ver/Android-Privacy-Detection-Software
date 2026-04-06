import matplotlib

matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib_zh import configure_matplotlib_chinese_font

configure_matplotlib_chinese_font()

# ==================== 数据定义 ====================
# 所有29个小点（与您的PPT完全对应）
all_points = [
    "适用的地域范围",
    "受规制的对象类型", 
    "受规制的数据活动",
    "排除适用的范围",
    "个人信息的定义",
    "敏感个人信息的处理规则",
    "未成年人个人信息",
    "死者个人信息",
    "匿名化、去识别化信息",
    "合法性基础",
    "同意规则",
    "数据本地化和出境安全评估",
    "跨境证据调取",
    "知情权",
    "访问权",
    "反对权",
    "删除权",
    "可携带权/复制权",
    "采取安全保障措施的义务",
    "保存（储存）期限",
    "发生数据安全事件时的通知义务",
    "个人信息保护影响评估",
    "DPO/个人信息保护责任人制度",
    "守门人条款",
    "自动化决策",
    "图像信息和身份识别信息采集",
    "民事诉讼",
    "行政监管",
    "刑事责任"
]

# 六个维度的分组（根据您的PPT结构）
dimension_groups = {
    "基础概念与适用范围": [0, 1, 2, 3, 4, 8],  # 6个点
    "处理的核心原则与合法性": [5, 6, 7, 9, 10, 19],  # 6个点
    "数据主体的权利": [13, 14, 15, 16, 17, 24],  # 6个点
    "处理者的义务与合规": [18, 20, 21, 22, 23, 25],  # 6个点
    "数据的跨境流动": [11, 12],  # 2个点
    "监管与法律责任": [26, 27, 28]  # 3个点
}

# 作用范围星级数据（PIPL, GDPR, CCPA）
scope_scores = {
    'PIPL': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 3, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4],
    'GDPR': [5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 5, 3, 5, 5, 4],
    'CCPA': [3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 2, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 2, 3, 1]
}

# 严格程度星级数据
strictness_scores = {
    'PIPL': [3, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 3, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4],
    'GDPR': [5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 5, 3, 5, 5, 4],
    'CCPA': [3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 2, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 2, 2, 1]
}

# 颜色定义
colors = {
    'PIPL': '#1f77b4',  # 蓝色
    'GDPR': '#ff7f0e',  # 橙色
    'CCPA': '#2ca02c'   # 绿色
}

# ==================== 改进的绘图函数 ====================
def plot_dimension_radar(dimension_name, point_indices, score_type="scope", figsize=(10, 8)):
    """
    绘制单个维度的雷达图
    """
    # 选择数据
    if score_type == "scope":
        data_dict = scope_scores
        title_suffix = "作用范围"
    else:
        data_dict = strictness_scores
        title_suffix = "严格程度"
    
    # 提取该维度的点
    dimension_points = [all_points[i] for i in point_indices]
    
    # 提取该维度的分数
    dimension_scores = {}
    for law in ['PIPL', 'GDPR', 'CCPA']:
        dimension_scores[law] = [data_dict[law][i] for i in point_indices]
    
    # 维度数量
    N = len(dimension_points)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # 绘制每个法律
    for law in ['PIPL', 'GDPR', 'CCPA']:
        values = dimension_scores[law]
        stats = values + values[:1]
        ax.plot(angles, stats, linewidth=3, label=law, color=colors[law])
        ax.fill(angles, stats, alpha=0.2, color=colors[law])
    
    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    
    # 根据点数调整标签字体大小
    if N <= 6:
        fontsize = 12
        rotation = 0
    elif N <= 10:
        fontsize = 10
        rotation = 0
    else:
        fontsize = 8
        rotation = 45
    
    # 设置标签，确保中文正确显示
    labels = dimension_points
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation)
    
    # 设置y轴
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)
    ax.set_rlabel_position(30)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 标题和图例
    title = f"{dimension_name} - {title_suffix}对比"
    plt.title(title, size=16, pad=25, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存文件
    filename = f"radar_{dimension_name}_{score_type}.png".replace(" ", "_").replace("/", "_")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成: {filename}")
    return filename

def plot_summary_radar(score_type="scope", figsize=(9, 7)):
    """
    绘制六个维度的汇总雷达图（每个维度的平均分）
    """
    if score_type == "scope":
        data_dict = scope_scores
        title_suffix = "作用范围"
    else:
        data_dict = strictness_scores
        title_suffix = "严格程度"
    
    # 计算每个维度的平均分
    dimension_names = list(dimension_groups.keys())
    dimension_avgs = {law: [] for law in ['PIPL', 'GDPR', 'CCPA']}
    
    for dim_name, indices in dimension_groups.items():
        for law in ['PIPL', 'GDPR', 'CCPA']:
            dim_scores = [data_dict[law][i] for i in indices]
            avg_score = np.mean(dim_scores)
            dimension_avgs[law].append(avg_score)
    
    # 维度数量
    N = len(dimension_names)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # 绘制每个法律
    for law in ['PIPL', 'GDPR', 'CCPA']:
        values = dimension_avgs[law]
        stats = values + values[:1]
        ax.plot(angles, stats, linewidth=3, marker='o', markersize=8, label=law, color=colors[law])
        ax.fill(angles, stats, alpha=0.15, color=colors[law])
    
    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimension_names, fontsize=11)
    
    # 设置y轴
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)
    ax.set_rlabel_position(30)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 标题和图例
    title = f"中美欧隐私法对比 - {title_suffix}星级（维度平均）"
    plt.title(title, size=16, pad=25, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存文件
    filename = f"radar_summary_{score_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成: {filename}")
    return filename

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("开始生成雷达图...")
    
    # 1. 为每个维度生成作用范围和严格程度雷达图
    dimension_filenames = {}
    for dim_name, indices in dimension_groups.items():
        print(f"\n生成 {dim_name} 的雷达图...")
        scope_file = plot_dimension_radar(dim_name, indices, "scope")
        strictness_file = plot_dimension_radar(dim_name, indices, "strictness")
        dimension_filenames[dim_name] = {"scope": scope_file, "strictness": strictness_file}
    
    # 2. 生成汇总雷达图
    print("\n生成汇总雷达图...")
    summary_scope = plot_summary_radar("scope")
    summary_strictness = plot_summary_radar("strictness")
    
    # 3. 生成所有29个点的完整雷达图（可选）
    print("\n生成完整29点雷达图...")
    plot_dimension_radar("完整29点", list(range(29)), "scope", figsize=(12, 10))
    plot_dimension_radar("完整29点", list(range(29)), "strictness", figsize=(12, 10))
    
    print("\n" + "="*50)
    print("所有雷达图已生成完成！")
    print("="*50)
    print("\n生成的文件列表：")
    for dim_name, files in dimension_filenames.items():
        print(f"  {dim_name}:")
        print(f"    作用范围: {files['scope']}")
        print(f"    严格程度: {files['strictness']}")
    print(f"\n  汇总图 - 作用范围: {summary_scope}")
    print(f"  汇总图 - 严格程度: {summary_strictness}")
    print(f"  完整29点 - 作用范围: radar_完整29点_scope.png")
    print(f"  完整29点 - 严格程度: radar_完整29点_strictness.png")
