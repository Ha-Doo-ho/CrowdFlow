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


HEATMAP_DISPLAY_WIDTH_PX = 640
TREND_CHART_HEIGHT_PX = 220


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

    st.title("🚁 CrowdFlow 실시간 밀집도 관제 시스템")
    st.caption("드론 조감 영상 기반 군중 밀집도 정량 측정 및 위험 구역 시각화")

    # ─── 모듈 초기화 ───
    renderer = HeatmapRenderer()
    db_path = os.path.join(PROJECT_ROOT, 'data', 'crowdflow.db')

    # JSON 경로 (main.py가 저장하는 파일)
    json_path = os.path.join(PROJECT_ROOT, 'data', 'latest_result.json')

    # ─── 레이아웃 ───
    col_main, col_side = st.columns([3, 1])

    # ─── 사이드바 (우측) ───
    with col_side:
        st.subheader("📊 현재 상태")

        # Level 설명
        st.markdown("""
        | Level | 상태 | 밀집도 |
        |-------|------|--------|
        | 🟢 1 | 안전 | < 2인/m² |
        | 🟡 2 | 주의 | 2~4인/m² |
        | 🟠 3 | 경고 | 4~6인/m² |
        | 🔴 4 | 위험 | 6~8인/m² |
        | 🟣 5 | 긴급 | > 8인/m² |
        """)

        total_frames_metric = st.empty()
        total_alerts_metric = st.empty()
        peak_density_metric = st.empty()
        avg_inference_metric = st.empty()
        ignored_metric = st.empty()
        predicted_metric = st.empty()
        cumulative_metric = st.empty()

    # ─── 메인 화면 (좌측) ───
    with col_main:
        # 히트맵 표시 영역
        heatmap_placeholder = st.empty()

        # 경보 표시 영역
        alert_placeholder = st.empty()

        # 상세 정보 영역
        info_placeholder = st.empty()

        # 추세 차트 영역
        chart_placeholder = st.empty()

    # ─── 실시간 갱신 루프 ───
    st.info("🔄 main.py 실행 후 자동으로 데이터가 표시됩니다. (1초 간격 갱신)")

    density_history = []
    last_timestamp = None
    stats_logger = None

    while True:
        result = load_latest_result(json_path)

        if result is not None:
            if os.path.exists(db_path):
                try:
                    if stats_logger is None:
                        stats_logger = DataLogger(db_path, verbose=False)
                    stats = stats_logger.get_stats()
                    total_frames_metric.metric(
                        "총 분석 프레임",
                        stats["total_frames"],
                    )
                    total_alerts_metric.metric(
                        "총 경고 발생",
                        stats["total_alerts"],
                    )
                    peak_density_metric.metric(
                        "최대 밀집도",
                        f"{stats['peak_density']:.2f} 인/m²",
                    )
                    avg_inference_metric.metric(
                        "평균 추론시간",
                        f"{stats['avg_inference_ms']:.1f} ms",
                    )
                    ignored_metric.metric(
                        "영역 밖 제외 탐지",
                        stats["ignored_detections"],
                    )
                except sqlite3.Error:
                    if stats_logger is not None:
                        stats_logger.close()
                        stats_logger = None
                    ignored_metric.info("DB 연결 대기 중...")

            predicted_risk = result.get("predicted_risk", {})
            if predicted_risk.get("enabled"):
                horizon = float(predicted_risk.get("horizon_seconds", 10.0))
                if predicted_risk.get("has_enough_history"):
                    predicted_metric.metric(
                        f"{horizon:.0f}초 예측 최대",
                        f"{predicted_risk.get('max_predicted_density', 0):.2f} 인/m²",
                    )
                else:
                    predicted_metric.info("예측 데이터 누적 중...")
                cumulative_metric.metric(
                    "누적 위험 점수",
                    f"{predicted_risk.get('max_cumulative_score', 0):.0f}/100",
                )
            else:
                predicted_metric.empty()
                cumulative_metric.empty()

            # ─── 히트맵 표시 ───
            with heatmap_placeholder.container():
                heatmap_image = renderer.render_to_bytes(result)
                st.image(heatmap_image, width=HEATMAP_DISPLAY_WIDTH_PX)

            # ─── 경보 표시 ───
            with alert_placeholder.container():
                alerts = result.get('alerts', [])
                max_level = int(result.get("max_level", result['level'].max()))

                if max_level >= 4:
                    st.error(f"🚨 긴급 경보! 최대 밀집도: {result['max_density']:.2f}인/m² | 경고 {len(alerts)}건")
                elif max_level >= 3:
                    st.warning(f"⚠️ 경고! 최대 밀집도: {result['max_density']:.2f}인/m² | 경고 {len(alerts)}건")
                elif max_level >= 2:
                    st.info(f"🔔 주의 구역 감지 | 최대 밀집도: {result['max_density']:.2f}인/m²")
                else:
                    st.success(f"✅ 안전 | 최대 밀집도: {result['max_density']:.2f}인/m²")

                # 개별 경고 표시
                if alerts:
                    for alert in alerts[:5]:  # 최대 5개만 표시
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
                        for cell in soon_cells[:3]:
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
                total = int(result['count'].sum())
                metadata = result.get("metadata", {})
                calibration_mode = metadata.get("calibration_mode", "unknown")
                model_name = metadata.get("model_name", "unknown")
                inference_ms = metadata.get("inference_ms", 0)
                ignored_count = int(result.get("ignored_count", 0))

                st.markdown(f"**시각:** {result['timestamp']} | "
                            f"**총 탐지:** {total}명 | "
                            f"**최대 밀집도:** {result['max_density']:.2f}인/m²")
                st.markdown(
                    f"**모델:** {model_name} | "
                    f"**추론:** {inference_ms:.1f}ms | "
                    f"**영역 밖 제외:** {ignored_count}건"
                )

                if predicted_risk.get("enabled"):
                    horizon = float(predicted_risk.get("horizon_seconds", 10.0))
                    if predicted_risk.get("has_enough_history"):
                        st.markdown(
                            f"**{horizon:.0f}초 예측 최대 밀집도:** "
                            f"{predicted_risk.get('max_predicted_density', 0):.2f}인/m² | "
                            f"**최대 증가율:** "
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
