# heatmap_renderer.py (강민재 담당)
# 두호의 grid_result를 받아서 히트맵 이미지를 생성하는 모듈
# 김대현의 dashboard.py에서 이 함수를 호출하여 화면에 표시

import matplotlib
matplotlib.use('Agg')  # Streamlit 호환 (GUI 백엔드 비활성화)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import io
import os

import numpy as np


def _configure_korean_font():
    font_path = r"C:\Windows\Fonts\malgun.ttf"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False


_configure_korean_font()


class HeatmapRenderer:
    def __init__(self, grid_size=2.0, area_width=10.0, area_height=10.0):
        self.grid_size = grid_size
        self.area_width = area_width
        self.area_height = area_height

        # Level별 색상 정의 (녹→황→주황→적→자주)
        self.level_colors = {
            0: '#CCCCCC',   # 데이터 없음 (회색)
            1: '#2ECC71',   # 안전 (녹색)
            2: '#F1C40F',   # 주의 (황색)
            3: '#E67E22',   # 경고 (주황)
            4: '#E74C3C',   # 위험 (적색)
            5: '#8E44AD',   # 긴급 (자주)
        }

        self.level_names = {
            0: '없음', 1: '안전', 2: '주의',
            3: '경고', 4: '위험', 5: '긴급'
        }

    def render(self, grid_result):
        """
        입력: grid_result dict (두호의 grid_calculator.py 출력)
        출력: matplotlib Figure 객체 → dashboard.py에서 st.pyplot(fig)로 표시

        ★ 이 함수가 프로젝트에서 가장 눈에 보이는 결과물을 만든다.
           경진대회 심사위원이 가장 먼저 보는 화면이 이 히트맵이다.
        """
        density = grid_result['grid']    # (rows, cols) 밀집도
        level = grid_result['level']     # (rows, cols) Level 1~5
        count = grid_result['count']     # (rows, cols) 인원수
        alerts = grid_result.get('alerts', [])
        risk_clusters = grid_result.get('risk_clusters', [])
        predicted_risk = grid_result.get('predicted_risk', {})
        grid_size = float(grid_result.get("grid_size", self.grid_size))
        area_width = float(grid_result.get("area_width", self.area_width))
        area_height = float(grid_result.get("area_height", self.area_height))

        rows, cols = density.shape

        # ─── Figure 생성 ───
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlim(0, area_width)
        ax.set_ylim(0, area_height)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title('CrowdFlow 실시간 밀집도 히트맵', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # 이미지 좌표계와 일치시키기 위해 Y축 반전

        # ─── 격자별 색상 칠하기 ───
        alert_cells = set()
        for a in alerts:
            alert_cells.add((a['row'], a['col']))

        for r in range(rows):
            for c in range(cols):
                x = c * grid_size
                y = r * grid_size
                lv = int(level[r, c])
                color = self.level_colors.get(lv, '#CCCCCC')

                # 격자 사각형 그리기
                rect = patches.Rectangle(
                    (x, y), grid_size, grid_size,
                    linewidth=1, edgecolor='white', facecolor=color, alpha=0.8
                )
                ax.add_patch(rect)

                # 격자 안에 밀집도 수치 표시
                d = density[r, c]
                cnt = int(count[r, c])
                if cnt > 0:
                    text_color = 'white' if lv >= 3 else 'black'
                    ax.text(x + grid_size/2, y + grid_size/2,
                            f'{d:.1f}\n({cnt}명)',
                            ha='center', va='center',
                            fontsize=8, fontweight='bold', color=text_color)

                # 경고 셀에 빗금 표시
                if (r, c) in alert_cells:
                    rect2 = patches.Rectangle(
                        (x, y), grid_size, grid_size,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--', hatch='///'
                    )
                    ax.add_patch(rect2)

        self._draw_risk_clusters(ax, risk_clusters)
        self._draw_predicted_risk(ax, predicted_risk, grid_size)

        # ─── 범례 ───
        legend_elements = []
        for lv in [1, 2, 3, 4, 5]:
            legend_elements.append(
                patches.Patch(facecolor=self.level_colors[lv],
                              edgecolor='gray',
                              label=f'Level {lv}: {self.level_names[lv]}')
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        return fig

    def _draw_risk_clusters(self, ax, risk_clusters):
        cluster_colors = {
            3: '#E67E22',
            4: '#E74C3C',
            5: '#8E44AD',
        }

        for cluster in risk_clusters:
            bbox = cluster.get('bbox_m', {})
            x1 = float(bbox.get('x1', 0.0))
            y1 = float(bbox.get('y1', 0.0))
            x2 = float(bbox.get('x2', x1))
            y2 = float(bbox.get('y2', y1))
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            max_level = int(cluster.get('max_level', 3))
            color = cluster_colors.get(max_level, '#E67E22')
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=color, facecolor=color,
                alpha=0.18, linestyle='-', zorder=5
            )
            ax.add_patch(rect)

            outline = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=color, facecolor='none',
                linestyle='-', zorder=6
            )
            ax.add_patch(outline)

            label = (
                f"위험구역 {cluster.get('id')} | "
                f"{cluster.get('max_density', 0):.1f}인/m² | "
                f"{cluster.get('total_count', 0)}명"
            )
            ax.text(
                x1 + 0.1, y1 + 0.25, label,
                ha='left', va='top',
                fontsize=8, fontweight='bold', color='white',
                bbox={'facecolor': color, 'alpha': 0.9, 'pad': 2, 'edgecolor': 'none'},
                zorder=7,
            )

    def _draw_predicted_risk(self, ax, predicted_risk, grid_size):
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
                x = col * grid_size
                y = row * grid_size
                rect = patches.Rectangle(
                    (x, y), grid_size, grid_size,
                    linewidth=2, edgecolor='#2C3E50', facecolor='none',
                    linestyle=':', zorder=8
                )
                ax.add_patch(rect)

        if not predicted_risk.get("has_enough_history"):
            return

        horizon = float(predicted_risk.get("horizon_seconds", 10.0))
        for cluster in predicted_risk.get("clusters", []):
            bbox = cluster.get('bbox_m', {})
            x1 = float(bbox.get('x1', 0.0))
            y1 = float(bbox.get('y1', 0.0))
            x2 = float(bbox.get('x2', x1))
            y2 = float(bbox.get('y2', y1))
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            color = '#00A8E8'
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2.5, edgecolor=color, facecolor='none',
                linestyle='--', zorder=9
            )
            ax.add_patch(rect)

            label = (
                f"{horizon:.0f}초 예측 {cluster.get('id')} | "
                f"{cluster.get('max_density', 0):.1f}인/m²"
            )
            ax.text(
                x1 + 0.1, y2 - 0.15, label,
                ha='left', va='bottom',
                fontsize=8, fontweight='bold', color='white',
                bbox={'facecolor': color, 'alpha': 0.9, 'pad': 2, 'edgecolor': 'none'},
                zorder=10,
            )

    def render_to_bytes(self, grid_result):
        """
        Figure를 PNG 바이트로 변환 (Streamlit st.image() 용)
        """
        fig = self.render(grid_result)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
