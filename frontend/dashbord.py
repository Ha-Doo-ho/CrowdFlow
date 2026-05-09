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
import sys
import time

# 프로젝트 루트를 import 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from frontend.heatmap_renderer import HeatmapRenderer
from frontend.data_logger import DataLogger


def load_latest_result(json_path):
    """
    main.py가 저장한 최신 분석 결과를 JSON에서 로드
    ★ main.py에서 매 프레임마다 이 파일을 덮어쓰고,
       dashboard.py에서 1초마다 이 파일을 읽는 구조
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # JSON → NumPy 변환
        data['grid'] = np.array(data['grid'])
        data['level'] = np.array(data['level'])
        data['count'] = np.array(data['count'])
        data['avg_conf'] = np.array(data['avg_conf'])
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

        # DB 통계 (있으면)
        if os.path.exists(db_path):
            try:
                logger = DataLogger(db_path)
                stats = logger.get_stats()
                st.metric("총 분석 프레임", stats['total_frames'])
                st.metric("총 경고 발생", stats['total_alerts'])
                st.metric("최대 밀집도", f"{stats['peak_density']:.2f} 인/m²")
                logger.close()
            except Exception:
                st.info("DB 연결 대기 중...")

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

    density_history = []  # 추세 차트용

    while True:
        result = load_latest_result(json_path)

        if result is not None:
            # ─── 히트맵 표시 ───
            with heatmap_placeholder.container():
                fig = renderer.render(result)
                st.pyplot(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)

            # ─── 경보 표시 ───
            with alert_placeholder.container():
                alerts = result.get('alerts', [])
                max_level = int(result['level'].max())

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

            # ─── 상세 정보 ───
            with info_placeholder.container():
                total = int(result['count'].sum())
                st.markdown(f"**시각:** {result['timestamp']} | "
                            f"**총 탐지:** {total}명 | "
                            f"**최대 밀집도:** {result['max_density']:.2f}인/m²")

            # ─── 추세 차트 ───
            density_history.append(result['max_density'])
            if len(density_history) > 30:
                density_history.pop(0)

            with chart_placeholder.container():
                st.subheader("📈 밀집도 추세 (최근 30 프레임)")
                st.line_chart(density_history)

        else:
            with heatmap_placeholder.container():
                st.warning("⏳ 데이터 대기 중... main.py를 먼저 실행해주세요.")

        time.sleep(1)  # 1초 간격 갱신


if __name__ == '__main__':
    main()
