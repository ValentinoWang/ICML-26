import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx

# 设置风格和字体
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"  # 接近 LaTeX 风格的数学字体
plt.rcParams["font.size"] = 12

def draw_figure():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # ==========================================
    # 1. 标题 (Bold Times New Roman)
    # ==========================================
    headers_y = 5.2
    ax.text(2.5, headers_y, "Input Analysis\n(Scatter Plot)", ha="center", va="center", fontweight="bold", fontsize=14)
    ax.text(7, headers_y, "Set Modeling", ha="center", va="center", fontweight="bold", fontsize=14)
    ax.text(11.5, headers_y, "Dual-head & Update", ha="center", va="center", fontweight="bold", fontsize=14)

    # ==========================================
    # 2. Input Analysis (Left)
    # ==========================================
    # 边框
    rect_scatter = patches.FancyBboxPatch((0.5, 0.5), 4, 4, boxstyle="round,pad=0.1", ec="black", fc="none", lw=1.5)
    ax.add_patch(rect_scatter)

    # 模拟数据点
    np.random.seed(42)
    # Noise (Grey)
    noise_x = np.random.uniform(0.8, 4.2, 10)
    noise_y = np.random.uniform(0.8, 4.2, 10)
    ax.scatter(noise_x, noise_y, c="grey", s=50, label="Noise", zorder=2)

    # Signal (Blue) - 聚在一起
    sig_x = np.random.normal(1.5, 0.2, 8)
    sig_y = np.random.normal(1.5, 0.2, 8)
    ax.scatter(sig_x, sig_y, c="#4472C4", s=60, label="Signal", zorder=3)  # Matplotlib default blue-ish

    # Drifted (Orange) - 移位
    drift_x = sig_x + 1.8
    drift_y = sig_y + 1.8
    ax.scatter(drift_x, drift_y, c="#ED7D31", s=60, label="Drifted", zorder=3)  # Matplotlib default orange-ish

    # 标注文字 (Times New Roman)
    ax.text(1.2, 2.5, "Signal", color="#4472C4", fontsize=12, fontweight="bold")
    ax.text(3.5, 3.8, "Drifted", color="#ED7D31", fontsize=12, fontweight="bold")
    ax.text(1.0, 3.8, "Noise", color="grey", fontsize=12, fontweight="bold")

    # Bias Arrow
    arrow = patches.FancyArrowPatch(
        (1.6, 1.6),
        (3.2, 3.2),
        arrowstyle="-|>,head_width=0.4,head_length=0.8",
        color="black",
        lw=2,
    )
    ax.add_patch(arrow)
    ax.text(2.1, 2.6, "bias drift\n" + r"$\mathbf{b}(e)$", ha="center", rotation=-45, fontsize=11)

    # ==========================================
    # 3. Projection & Flow
    # ==========================================
    # 梯形 Projection
    trap_coords = [[4.7, 2.5], [5.3, 3.0], [5.3, 2.0], [4.7, 2.5]]  # Simplified shape logic
    # 画一个梯形表示投影
    polygon = patches.Polygon([[4.8, 2.2], [4.8, 2.8], [5.4, 3.2], [5.4, 1.8]], closed=True, ec="black", fc="white", lw=1)
    ax.add_patch(polygon)
    # 内部线条表示矩阵变换
    ax.plot([4.9, 5.3], [2.3, 1.9], color="lightgrey", lw=0.5)
    ax.plot([4.9, 5.3], [2.5, 2.5], color="lightgrey", lw=0.5)
    ax.plot([4.9, 5.3], [2.7, 3.1], color="lightgrey", lw=0.5)

    ax.annotate("", xy=(5.5, 2.5), xytext=(4.5, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5.0, 3.4, "Project:\n" + r"$\mathbf{z}_i = U^T \Delta \theta_i$", ha="center", fontsize=11)

    # ==========================================
    # 4. Set Modeling (Middle) - 去掉 Attention 白框
    # ==========================================
    # 大边框
    rect_set = patches.FancyBboxPatch((5.8, 0.5), 2.5, 4, boxstyle="round,pad=0.1", ec="black", fc="none", lw=1.5)
    ax.add_patch(rect_set)

    ax.text(7.05, 3.6, "Set\nTransformer\n(SAB $\\times$ L, PMA)", ha="center", va="center", fontsize=12)

    # 网络拓扑图 (直接画在背景上，无白框)
    # 创建一个小图
    G = nx.random_geometric_graph(8, 0.6, seed=42)
    pos = nx.spring_layout(G, center=(7.05, 1.8), scale=0.6)

    # 绘制边 (深灰色，变细)
    for (u, v) in G.edges():
        x_vals = [pos[u][0], pos[v][0]]
        y_vals = [pos[u][1], pos[v][1]]
        ax.plot(x_vals, y_vals, color="#555555", alpha=0.6, lw=0.8, zorder=1)

    # 绘制点 (蓝橙混合)
    node_colors = ["#4472C4"] * 4 + ["#ED7D31"] * 4
    for i, (node, coords) in enumerate(pos.items()):
        circle = plt.Circle(coords, 0.08, color=node_colors[i], zorder=2)
        ax.add_patch(circle)

    # ==========================================
    # 5. Dual-head & Update (Right)
    # ==========================================
    # Head 1 Box
    rect_h1 = patches.FancyBboxPatch((9.2, 3.0), 2.0, 1.2, boxstyle="round,pad=0.05", ec="black", fc="white", lw=1.2)
    ax.add_patch(rect_h1)
    ax.text(10.2, 3.6, "Head 1:\nReweighting\n(variance control)", ha="center", va="center", fontsize=11)

    # Head 2 Box
    rect_h2 = patches.FancyBboxPatch((9.2, 0.8), 2.0, 1.2, boxstyle="round,pad=0.05", ec="black", fc="white", lw=1.2)
    ax.add_patch(rect_h2)
    ax.text(10.2, 1.4, "Head 2:\nCorrection\n(bias subtraction)", ha="center", va="center", fontsize=11)

    # Update Box (Formula)
    rect_upd = patches.FancyBboxPatch((11.8, 2.2), 2.1, 0.6, boxstyle="round,pad=0.05", ec="black", fc="none", lw=1.2)
    ax.add_patch(rect_upd)
    ax.text(
        12.85,
        2.5,
        r"$\theta_{t+1} = \theta_t + \Delta \theta_{\mathrm{weighted}} - \eta \Delta \phi$",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Arrows (Blue & Orange)
    # Split arrow
    ax.plot([8.3, 8.8], [2.5, 2.5], color="black", lw=1.5)  # main stem
    # to head 1
    ax.annotate("", xy=(9.2, 3.6), xytext=(8.8, 3.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.plot([8.8, 8.8], [2.5, 3.6], color="black", lw=1.5)
    # to head 2
    ax.annotate("", xy=(9.2, 1.4), xytext=(8.8, 1.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.plot([8.8, 8.8], [2.5, 1.4], color="black", lw=1.5)

    # Blue path
    ax.annotate("", xy=(12.8, 2.8), xytext=(11.2, 2.8), arrowprops=dict(arrowstyle="->", color="#4472C4", lw=2))
    ax.plot([11.2, 12.8], [3.6, 3.6], color="#4472C4", lw=2)  # out of box
    ax.plot([12.8, 12.8], [3.6, 2.8], color="#4472C4", lw=2)  # down to formula
    ax.text(12.0, 4.0, "weighted\nupdate\n" + r"$\Delta \theta_{\mathrm{weighted}}$", color="#4472C4", ha="center", fontsize=10)
    ax.text(11.4, 3.8, r"$w_i$", color="#4472C4", ha="center", fontsize=11, style="italic")

    # Orange path
    ax.annotate("", xy=(12.8, 2.2), xytext=(11.2, 2.2), arrowprops=dict(arrowstyle="->", color="#ED7D31", lw=2))
    ax.plot([11.2, 12.8], [1.4, 1.4], color="#ED7D31", lw=2)  # out of box
    ax.plot([12.8, 12.8], [1.4, 2.2], color="#ED7D31", lw=2)  # up to formula
    ax.text(12.0, 1.0, "back-project:\n" + r"$\Delta \phi = U \Delta \phi_{\mathrm{emb}}$", ha="center", fontsize=10)
    ax.text(12.3, 1.8, "Est. Bias\n" + r"$\approx \mathbf{b}(e)$", color="#ED7D31", ha="center", fontsize=10)

    # ==========================================
    # 6. Bottom Loss Formula
    # ==========================================
    # Manual color composition for the formula
    formula_y = 0.05
    # Part 1: Black
    t1 = ax.text(
        6.0,
        formula_y,
        r"$\mathcal{L} = \mathbb{E}\left[||\theta_{t+1} - \theta^*||^2\right] + \lambda \cdot$",
        ha="right",
        fontsize=13,
    )
    # Part 2: Blue
    t2 = ax.text(6.0, formula_y, r" $\mathcal{L}_{\mathrm{weight}}$", color="#4472C4", ha="left", fontsize=13)
    # Part 3: Black
    t3 = ax.text(7.2, formula_y, r"$+ \mu \cdot$", color="black", ha="left", fontsize=13)
    # Part 4: Orange
    t4 = ax.text(8.0, formula_y, r" $\mathcal{L}_{\mathrm{corr}}$", color="#ED7D31", ha="left", fontsize=13)

    plt.tight_layout()
    # Save implies the user can run this.
    # For display in chat, we just show it.
    plt.show()

draw_figure()
