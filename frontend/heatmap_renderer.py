import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def _configure_korean_font():
    font_path = r"C:\Windows\Fonts\malgun.ttf"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


_configure_korean_font()


class HeatmapRenderer:
    def __init__(self, grid_size=2.0, area_width=10.0, area_height=10.0):
        self.grid_size = grid_size
        self.area_width = area_width
        self.area_height = area_height

        self.level_colors = {
            0: "#CCCCCC",
            1: "#2ECC71",
            2: "#F1C40F",
            3: "#E67E22",
            4: "#E74C3C",
            5: "#8E44AD",
        }
        self.level_names = {
            0: "없음",
            1: "안전",
            2: "주의",
            3: "경고",
            4: "위험",
            5: "긴급",
        }

    def render(self, grid_result):
        density = np.asarray(grid_result["grid"], dtype=float)
        level = np.asarray(grid_result["level"])
        count = np.asarray(grid_result["count"])
        alerts = grid_result.get("alerts", [])
        risk_clusters = grid_result.get("risk_clusters", [])
        predicted_risk = grid_result.get("predicted_risk", {})
        grid_size = float(grid_result.get("grid_size", self.grid_size))
        area_width = float(grid_result.get("area_width", self.area_width))
        area_height = float(grid_result.get("area_height", self.area_height))

        rows, cols = density.shape
        x_edges, y_edges = self._grid_edges(grid_result, rows, cols, grid_size,
                                            area_width, area_height)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlim(0, area_width)
        ax.set_ylim(0, area_height)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_title("CrowdFlow 실시간 밀집도 히트맵", fontsize=14, fontweight="bold")
        ax.invert_yaxis()

        alert_cells = {(int(a["row"]), int(a["col"])) for a in alerts}

        for row in range(rows):
            for col in range(cols):
                x1 = float(x_edges[col])
                y1 = float(y_edges[row])
                width = float(x_edges[col + 1] - x1)
                height = float(y_edges[row + 1] - y1)
                lv = int(level[row, col])
                color = self.level_colors.get(lv, "#CCCCCC")

                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=1, edgecolor="white", facecolor=color, alpha=0.8
                )
                ax.add_patch(rect)

                cell_count = int(count[row, col])
                if cell_count > 0:
                    text_color = "white" if lv >= 3 else "black"
                    ax.text(
                        x1 + width / 2,
                        y1 + height / 2,
                        f"{density[row, col]:.1f}\n({cell_count}명)",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=text_color,
                    )

                if (row, col) in alert_cells:
                    alert_rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor="red", facecolor="none",
                        linestyle="--", hatch="///"
                    )
                    ax.add_patch(alert_rect)

        self._draw_risk_clusters(ax, risk_clusters)
        self._draw_predicted_risk(ax, predicted_risk, grid_size, x_edges, y_edges)

        legend_elements = [
            patches.Patch(
                facecolor=self.level_colors[lv],
                edgecolor="gray",
                label=f"Level {lv}: {self.level_names[lv]}",
            )
            for lv in [1, 2, 3, 4, 5]
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.tight_layout()
        return fig

    def _grid_edges(self, grid_result, rows, cols, grid_size, area_width, area_height):
        x_edges = grid_result.get("x_edges")
        y_edges = grid_result.get("y_edges")
        if x_edges is None:
            x_edges = [min(col * grid_size, area_width) for col in range(cols + 1)]
            if area_width < cols * grid_size:
                x_edges[-1] = area_width
        if y_edges is None:
            y_edges = [min(row * grid_size, area_height) for row in range(rows + 1)]
            if area_height < rows * grid_size:
                y_edges[-1] = area_height
        return np.asarray(x_edges, dtype=float), np.asarray(y_edges, dtype=float)

    def _draw_risk_clusters(self, ax, risk_clusters):
        cluster_colors = {
            3: "#E67E22",
            4: "#E74C3C",
            5: "#8E44AD",
        }

        for cluster in risk_clusters:
            bbox = cluster.get("bbox_m", {})
            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", x1))
            y2 = float(bbox.get("y2", y1))
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            max_level = int(cluster.get("max_level", 3))
            color = cluster_colors.get(max_level, "#E67E22")
            fill = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=color, facecolor=color,
                alpha=0.18, linestyle="-", zorder=5
            )
            ax.add_patch(fill)

            outline = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=color, facecolor="none",
                linestyle="-", zorder=6
            )
            ax.add_patch(outline)

            label = (
                f"위험구역 {cluster.get('id')} | "
                f"{cluster.get('max_density', 0):.1f}명/m^2 | "
                f"{cluster.get('total_count', 0)}명"
            )
            ax.text(
                x1 + 0.1, y1 + 0.25, label,
                ha="left", va="top",
                fontsize=8, fontweight="bold", color="white",
                bbox={"facecolor": color, "alpha": 0.9, "pad": 2, "edgecolor": "none"},
                zorder=7,
            )

    def _draw_predicted_risk(self, ax, predicted_risk, grid_size, x_edges, y_edges):
        if not predicted_risk or not predicted_risk.get("enabled"):
            return

        cumulative_score = predicted_risk.get("cumulative_score")
        if cumulative_score is not None:
            cumulative_score = np.asarray(cumulative_score, dtype=float)
            for cell in predicted_risk.get("cumulative_cells", [])[:5]:
                row = int(cell.get("row", 0))
                col = int(cell.get("col", 0))
                if not (0 <= row < cumulative_score.shape[0]
                        and 0 <= col < cumulative_score.shape[1]):
                    continue
                x1 = float(x_edges[col])
                y1 = float(y_edges[row])
                width = float(x_edges[col + 1] - x1)
                height = float(y_edges[row + 1] - y1)
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor="#2C3E50", facecolor="none",
                    linestyle=":", zorder=8
                )
                ax.add_patch(rect)

        if not predicted_risk.get("has_enough_history"):
            return

        horizon = float(predicted_risk.get("horizon_seconds", 10.0))
        for cluster in predicted_risk.get("clusters", []):
            bbox = cluster.get("bbox_m", {})
            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", x1))
            y2 = float(bbox.get("y2", y1))
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            color = "#00A8E8"
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2.5, edgecolor=color, facecolor="none",
                linestyle="--", zorder=9
            )
            ax.add_patch(rect)

            label = (
                f"{horizon:.0f}초 예측 {cluster.get('id')} | "
                f"{cluster.get('max_density', 0):.1f}명/m^2"
            )
            ax.text(
                x1 + 0.1, y2 - 0.15, label,
                ha="left", va="bottom",
                fontsize=8, fontweight="bold", color="white",
                bbox={"facecolor": color, "alpha": 0.9, "pad": 2, "edgecolor": "none"},
                zorder=10,
            )

    def render_to_bytes(self, grid_result):
        fig = self.render(grid_result)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
