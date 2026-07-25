# dashboard.py (김대현 담당)
# Streamlit 기반 실시간 관제 대시보드
# 실행: streamlit run frontend/dashboard.py

# ★ 이 파일은 main.py와 별도로 실행됨
# ★ main.py가 분석 결과를 JSON 파일로 저장 → dashboard.py가 읽어서 표시
# ★ 또는 SQLite DB에서 읽어서 표시

import streamlit as st
import numpy as np
import json
import os
import sqlite3
import sys
import time
from datetime import datetime

# 프로젝트 루트를 import 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from frontend.heatmap_renderer import HeatmapRenderer
from frontend.data_logger import DataLogger


TREND_CHART_HEIGHT_PX = 220


LEVEL_LEGEND_HTML = """
<div class="level-legend" aria-label="밀집도 단계 범례">
  <div class="legend-item"><span class="legend-dot level-1"></span><b>1 안전</b><span>&lt; 2인/m²</span></div>
  <div class="legend-item"><span class="legend-dot level-2"></span><b>2 주의</b><span>2~4인/m²</span></div>
  <div class="legend-item"><span class="legend-dot level-3"></span><b>3 경고</b><span>4~6인/m²</span></div>
  <div class="legend-item"><span class="legend-dot level-4"></span><b>4 위험</b><span>6~8인/m²</span></div>
  <div class="legend-item"><span class="legend-dot level-5"></span><b>5 긴급</b><span>&gt; 8인/m²</span></div>
</div>
"""


PAGE_STYLE = """
<style>
  .block-container {
    max-width: 1120px;
    padding-top: 1rem;
    padding-bottom: 1.5rem;
  }
  h1 {
    font-size: 1.8rem !important;
    margin-bottom: 0.15rem !important;
  }
  [data-testid="stCaptionContainer"] {
    margin-bottom: 0.35rem;
  }
  [data-testid="stMetric"] {
    min-height: 74px;
    padding: 0.55rem 0.65rem;
    border: 1px solid rgba(128, 139, 154, 0.32);
    border-radius: 6px;
    background: rgba(35, 40, 49, 0.38);
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.76rem;
    line-height: 1.2;
  }
  [data-testid="stMetricValue"] {
    font-size: 1.18rem;
    line-height: 1.25;
  }
  .level-legend {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 6px;
    margin: 0.35rem 0 0.7rem;
  }
  .legend-item {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    min-height: 34px;
    padding: 5px 7px;
    border: 1px solid rgba(128, 139, 154, 0.32);
    border-radius: 4px;
    font-size: 0.74rem;
    white-space: nowrap;
  }
  .legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex: 0 0 10px;
  }
  .level-1 { background: #2ECC71; }
  .level-2 { background: #F1C40F; }
  .level-3 { background: #E67E22; }
  .level-4 { background: #E74C3C; }
  .level-5 { background: #8E44AD; }
  @media (max-width: 900px) {
    .level-legend { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .legend-item { white-space: normal; }
  }
</style>
"""


def load_latest_result(json_path):
    """
    main.py가 저장한 최신 분석 결과를 JSON에서 로드
    ★ main.py에서 매 프레임마다 이 파일을 덮어쓰고,
       dashboard.py에서 1초마다 이 파일을 읽는 구조
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # JSON → NumPy 변환
        data['grid'] = np.array(data['grid'])
        data['level'] = np.array(data['level'])
        data['count'] = np.array(data['count'])
        data['avg_conf'] = np.array(data['avg_conf'])
        predicted_risk = data.get("predicted_risk")
        if isinstance(predicted_risk, dict):
            for key in (
                "predicted_grid",
                "predicted_level",
                "growth_grid",
                "cumulative_score",
            ):
                if key in predicted_risk:
                    predicted_risk[key] = np.array(predicted_risk[key])
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    # ─── 페이지 설정 ───
    st.set_page_config(
        page_title="CrowdFlow 관제 시스템",
        page_icon="🚁",
        layout="wide"
    )

    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    st.title("CrowdFlow 실시간 밀집도 관제 시스템")
    st.caption("드론 조감 영상 기반 군중 밀집도 정량 측정 및 위험 구역 시각화")
    st.markdown(LEVEL_LEGEND_HTML, unsafe_allow_html=True)

    # ─── 모듈 초기화 ───
    renderer = HeatmapRenderer()
    db_path = os.path.join(PROJECT_ROOT, 'data', 'crowdflow.db')

    # JSON 경로 (main.py가 저장하는 파일)
    json_path = os.path.join(PROJECT_ROOT, 'data', 'latest_result.json')

    # ─── 상단 요약 지표 ───
    metrics_placeholder = st.empty()

    # ─── 메인 화면 ───
    col_heatmap, col_status = st.columns([1.7, 1], gap="small")

    with col_heatmap:
        heatmap_placeholder = st.empty()

    with col_status:
        status_placeholder = st.empty()

        alert_placeholder = st.empty()
        info_placeholder = st.empty()

    with st.expander("밀집도 추세와 상세 기록", expanded=False):
        chart_placeholder = st.empty()

    density_history = []
    last_timestamp = None
    stats_logger = None

    while True:
        result = load_latest_result(json_path)

        if result is not None:
            stats = {
                "total_frames": 0,
                "total_alerts": 0,
                "peak_density": 0.0,
                "ignored_detections": 0,
                "avg_inference_ms": 0.0,
            }
            if os.path.exists(db_path):
                try:
                    if stats_logger is None:
                        stats_logger = DataLogger(db_path, verbose=False)
                    stats = stats_logger.get_stats()
                except sqlite3.Error:
                    if stats_logger is not None:
                        stats_logger.close()
                        stats_logger = None

            predicted_risk = result.get("predicted_risk", {})
            metadata = result.get("metadata", {})
            current_total = int(result["count"].sum())
            current_ignored = int(result.get("ignored_count", 0))
            inference_ms = float(metadata.get("inference_ms", 0))
            horizon = float(predicted_risk.get("horizon_seconds", 10.0))
            if predicted_risk.get("enabled") and predicted_risk.get("has_enough_history"):
                predicted_value = (
                    f"{predicted_risk.get('max_predicted_density', 0):.2f} 인/m²"
                )
            elif predicted_risk.get("enabled"):
                predicted_value = "누적 중"
            else:
                predicted_value = "사용 안 함"
            cumulative_value = (
                f"{predicted_risk.get('max_cumulative_score', 0):.0f}/100"
                if predicted_risk.get("enabled") else "-"
            )

            with metrics_placeholder.container():
                current_metrics = st.columns(6, gap="small")
                current_metrics[0].metric("현재 탐지", f"{current_total}명")
                current_metrics[1].metric(
                    "현재 최대 밀집도",
                    f"{result['max_density']:.2f} 인/m²",
                )
                current_metrics[2].metric("현재 추론", f"{inference_ms:.1f} ms")
                current_metrics[3].metric("현재 영역 밖 제외", current_ignored)
                current_metrics[4].metric(f"{horizon:.0f}초 예측 최대", predicted_value)
                current_metrics[5].metric("누적 위험 점수", cumulative_value)

                history_metrics = st.columns(4, gap="small")
                history_metrics[0].metric("누적 분석 프레임", stats["total_frames"])
                history_metrics[1].metric("누적 경고", stats["total_alerts"])
                history_metrics[2].metric(
                    "역대 최대 밀집도",
                    f"{stats['peak_density']:.2f} 인/m²",
                )
                history_metrics[3].metric(
                    "누적 영역 밖 제외",
                    stats["ignored_detections"],
                )

            # ─── 히트맵 표시 ───
            with heatmap_placeholder.container():
                heatmap_image = renderer.render_to_bytes(result)
                st.image(heatmap_image, use_container_width=True)

            alerts = result.get('alerts', [])
            max_level = int(result.get("max_level", result['level'].max()))

            # ─── 현재 상태 ───
            with status_placeholder.container():
                if max_level >= 4:
                    st.error(f"🚨 긴급 경보! 최대 밀집도: {result['max_density']:.2f}인/m² | 경고 {len(alerts)}건")
                elif max_level >= 3:
                    st.warning(f"⚠️ 경고! 최대 밀집도: {result['max_density']:.2f}인/m² | 경고 {len(alerts)}건")
                elif max_level >= 2:
                    st.info(f"🔔 주의 구역 감지 | 최대 밀집도: {result['max_density']:.2f}인/m²")
                else:
                    st.success(f"✅ 안전 | 최대 밀집도: {result['max_density']:.2f}인/m²")

            # ─── 경보 표시 ───
            with alert_placeholder.container():
                # 개별 경고 표시
                if alerts:
                    for alert in alerts[:3]:
                        st.warning(f"→ {alert['message']}")

                if predicted_risk.get("enabled"):
                    soon_cells = predicted_risk.get("soon_risk_cells", [])
                    cumulative_cells = predicted_risk.get("cumulative_cells", [])
                    horizon = float(predicted_risk.get("horizon_seconds", 10.0))

                    if soon_cells:
                        st.warning(
                            f"⏱ {horizon:.0f}초 내 위험 가능 구역 "
                            f"{len(soon_cells)}건"
                        )
                        for cell in soon_cells[:2]:
                            st.info(
                                f"→ [{cell['row']},{cell['col']}] "
                                f"{cell['current_density']:.1f} → "
                                f"{cell['predicted_density']:.1f}인/m² 예상"
                            )

                    if cumulative_cells:
                        top_cell = cumulative_cells[0]
                        st.info(
                            "📌 반복 위험 구역 "
                            f"[{top_cell['row']},{top_cell['col']}] "
                            f"누적 점수 {top_cell['score']:.0f}/100"
                        )

            # ─── 상세 정보 ───
            with info_placeholder.container():
                calibration_mode = metadata.get("calibration_mode", "unknown")
                model_name = metadata.get("model_name", "unknown")

                st.markdown("#### 현재 프레임")
                st.caption(result['timestamp'])
                st.markdown(
                    f"**모델** {model_name}  \n"
                    f"**탐지** {current_total}명  \n"
                    f"**영역 밖 제외** {current_ignored}건"
                )

                if predicted_risk.get("enabled"):
                    if predicted_risk.get("has_enough_history"):
                        st.markdown(
                            f"**최대 증가율** "
                            f"{predicted_risk.get('max_growth_per_second', 0):.2f}인/m²/s"
                        )
                    else:
                        st.caption("예측 위험도는 최근 프레임을 누적한 뒤 표시됩니다.")

                if calibration_mode == "full_frame_fallback":
                    st.warning(
                        "임시 전체 프레임 매핑을 사용 중입니다. "
                        "현재 밀집도는 실측 검증값이 아닙니다."
                    )

                try:
                    result_time = datetime.fromisoformat(result["timestamp"])
                    age_seconds = (datetime.now() - result_time).total_seconds()
                    if age_seconds > 5:
                        st.warning(f"최근 분석 결과가 {age_seconds:.0f}초 전 데이터입니다.")
                except (TypeError, ValueError):
                    pass

            # ─── 추세 차트 ───
            if result["timestamp"] != last_timestamp:
                density_history.append(result['max_density'])
                last_timestamp = result["timestamp"]
                if len(density_history) > 30:
                    density_history.pop(0)

            with chart_placeholder.container():
                st.subheader("📈 밀집도 추세 (최근 30 프레임)")
                st.line_chart(density_history, height=TREND_CHART_HEIGHT_PX)

        else:
            with heatmap_placeholder.container():
                st.warning("⏳ 데이터 대기 중... main.py를 먼저 실행해주세요.")

        time.sleep(1)  # 1초 간격 갱신


if __name__ == '__main__':
    main()
