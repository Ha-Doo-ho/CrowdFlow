# data_logger.py (강민재 담당)
# 두호의 grid_result를 SQLite 데이터베이스에 기록하는 모듈
# 교내 실험 시 자동으로 데이터를 수집하고, 나중에 성능 평가에 사용

import sqlite3
import json
import os
import math
from datetime import datetime


class DataLogger:
    def __init__(self, db_path='data/crowdflow.db', verbose=True):
        """
        db_path: SQLite DB 파일 경로
        ★ SQLite는 별도 서버 불필요 — 파일 1개가 곧 DB
        """
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.conn = sqlite3.connect(db_path, timeout=5)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.cursor = self.conn.cursor()
        self.verbose = verbose
        self._create_tables()
        if self.verbose:
            print(f"[DB] 연결 완료: {db_path}")

    def _create_tables(self):
        """DB 테이블 생성 (최초 1회만 실행됨)"""

        # 프레임별 분석 결과 기록
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_detected INTEGER,
                max_density REAL,
                alert_count INTEGER,
                grid_data TEXT,
                level_data TEXT,
                count_data TEXT,
                conf_data TEXT,
                frame_number INTEGER,
                ignored_count INTEGER DEFAULT 0,
                inference_ms REAL,
                model_name TEXT,
                model_type TEXT,
                calibration_mode TEXT
            )
        """)

        # 경고 기록
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                cell_row INTEGER,
                cell_col INTEGER,
                avg_conf REAL,
                detected_count INTEGER,
                alert_type TEXT,
                message TEXT,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
            )
        """)

        # 교내 실험 기록 (강민재가 실험 자동화에 사용)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                exp_id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario INTEGER,
                actual_count INTEGER,
                estimated_count INTEGER,
                density_gt REAL,
                density_est REAL,
                area_m2 REAL,
                timestamp TEXT
            )
        """)

        self._ensure_column("frames", "frame_number", "INTEGER")
        self._ensure_column("frames", "ignored_count", "INTEGER DEFAULT 0")
        self._ensure_column("frames", "inference_ms", "REAL")
        self._ensure_column("frames", "model_name", "TEXT")
        self._ensure_column("frames", "model_type", "TEXT")
        self._ensure_column("frames", "calibration_mode", "TEXT")
        self._ensure_column("alerts", "alert_type", "TEXT")
        self.conn.commit()

    def _ensure_column(self, table_name, column_name, definition):
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in self.cursor.fetchall()}
        if column_name not in columns:
            self.cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )

    def log_frame(self, grid_result):
        """
        매 분석 프레임마다 호출 — main.py의 while 루프에서 사용

        입력: grid_result dict (두호의 grid_calculator.py 출력)
        ★ NumPy 배열은 JSON으로 직접 변환 안 됨 → .tolist()로 변환
        """
        metadata = grid_result.get("metadata", {})

        self.cursor.execute("""
            INSERT INTO frames
            (timestamp, total_detected, max_density, alert_count,
             grid_data, level_data, count_data, conf_data,
             frame_number, ignored_count, inference_ms,
             model_name, model_type, calibration_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            grid_result['timestamp'],
            int(grid_result['count'].sum()),
            float(grid_result['max_density']),
            len(grid_result.get('alerts', [])),
            json.dumps(grid_result['grid'].tolist()),
            json.dumps(grid_result['level'].tolist()),
            json.dumps(grid_result['count'].tolist()),
            json.dumps(grid_result['avg_conf'].tolist()),
            metadata.get("frame_number"),
            int(grid_result.get("ignored_count", 0)),
            metadata.get("inference_ms"),
            metadata.get("model_name"),
            metadata.get("model_type"),
            metadata.get("calibration_mode"),
        ))

        frame_id = self.cursor.lastrowid

        # 경고 기록
        for alert in grid_result.get('alerts', []):
            self.cursor.execute("""
                INSERT INTO alerts
                (frame_id, cell_row, cell_col, avg_conf,
                 detected_count, alert_type, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                frame_id,
                alert['row'], alert['col'],
                alert['avg_conf'], alert['count'],
                alert.get("type", "unknown"),
                alert['message']
            ))

        self.conn.commit()
        return frame_id

    def log_experiment(self, scenario, actual_count, estimated_count, area_m2):
        """
        교내 실험 시 Ground Truth 기록 (강민재가 현장에서 사용)

        scenario: 시나리오 번호 (1~5)
        actual_count: 실제 인원 수 (사전에 기록한 값)
        estimated_count: 시스템이 추정한 인원 수
        area_m2: 배치 면적 (m²)
        """
        density_gt = actual_count / area_m2 if area_m2 > 0 else 0
        density_est = estimated_count / area_m2 if area_m2 > 0 else 0

        self.cursor.execute("""
            INSERT INTO experiments
            (scenario, actual_count, estimated_count,
             density_gt, density_est, area_m2, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            scenario, actual_count, estimated_count,
            density_gt, density_est, area_m2,
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def get_recent_frames(self, n=10):
        """최근 n개 프레임 데이터 조회 (대시보드 추세 차트용)"""
        self.cursor.execute("""
            SELECT timestamp, total_detected, max_density, alert_count
            FROM frames
            ORDER BY frame_id DESC
            LIMIT ?
        """, (n,))
        rows = self.cursor.fetchall()
        rows.reverse()  # 시간순 정렬
        return rows

    def get_stats(self):
        """전체 통계 (대시보드 사이드바용)"""
        self.cursor.execute("SELECT COUNT(*) FROM frames")
        total_frames = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM alerts")
        total_alerts = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT MAX(max_density) FROM frames")
        peak_density = self.cursor.fetchone()[0] or 0

        self.cursor.execute("SELECT SUM(ignored_count) FROM frames")
        ignored_detections = self.cursor.fetchone()[0] or 0

        self.cursor.execute(
            "SELECT AVG(inference_ms) FROM frames WHERE inference_ms IS NOT NULL"
        )
        avg_inference_ms = self.cursor.fetchone()[0] or 0

        return {
            'total_frames': total_frames,
            'total_alerts': total_alerts,
            'peak_density': peak_density,
            'ignored_detections': ignored_detections,
            'avg_inference_ms': avg_inference_ms,
        }

    def get_experiment_metrics(self):
        self.cursor.execute("""
            SELECT actual_count, estimated_count, density_gt, density_est
            FROM experiments
        """)
        rows = self.cursor.fetchall()
        if not rows:
            return {
                "sample_count": 0,
                "count_mae": 0.0,
                "count_rmse": 0.0,
                "density_mae": 0.0,
                "density_rmse": 0.0,
            }

        count_errors = [estimated - actual for actual, estimated, _, _ in rows]
        density_errors = [
            estimated_density - actual_density
            for _, _, actual_density, estimated_density in rows
        ]

        return {
            "sample_count": len(rows),
            "count_mae": sum(abs(error) for error in count_errors) / len(rows),
            "count_rmse": math.sqrt(
                sum(error ** 2 for error in count_errors) / len(rows)
            ),
            "density_mae": (
                sum(abs(error) for error in density_errors) / len(rows)
            ),
            "density_rmse": math.sqrt(
                sum(error ** 2 for error in density_errors) / len(rows)
            ),
        }

    def close(self):
        """DB 연결 종료"""
        self.conn.close()
        if self.verbose:
            print("[DB] 연결 종료")
