# -*- coding: utf-8 -*-
"""
"엔하2 터치 + 피보4/피보5 국소 반등 전환(피보턴)이 근접해서 겹치는 경우"를
새 매수 신호로 검증하고, 기존에 이미 검증해둔 엔하3(721건, 손익비1.15~1.40)/
엔하4(374건, 1.20~1.50)/엔하5(151건, 1.51~1.74)와 나란히 비교하는 실행
스크립트다.

지표/신호 계산은 전부 재사용한다:
  - 엔상/엔하/피보4·5 계산: src/reversal_angle.py의 add_all_indicators
  - 피보턴 판정 + 엔하2와의 거리 계산: src/enha2_fibo_turn.py (직전 세션의
    src/triple_indicator_confluence.py 함수를 그대로 가져다 씀)
  - 매매 시뮬레이션: src/reversal_angle.py의 simulate_reversal_trades
  - 승률/손익비/학습검증분리: src/ma50_backtest.py
  - 엔하3/4/5 비교 기준: 재계산하지 않고 기존 결과 파일
    data/reversal_angle_enhah_3_4_5_comparison.csv를 그대로 읽어서 씀
    (run_reversal_angle_enha_3_4_5_comparison.py가 엔하5를 재사용한 것과
    같은 이유 - 재계산으로 인한 숫자 미세 차이를 피하기 위함)
  - 실전비용 가정: run_enha_ensah_full_matrix.py와 동일(왕복 0.38%p)

검증 절차:
  1. 엔하2 터치 전체 건수 vs 피보턴 근접 필터를 통과한 최종 건수
  2. 최종 신호의 승률/손익비 (학습/검증 분리)
  3. 엔하3/4/5 대비 표본수·승률·손익비 비교표
  4. 실전비용(왕복 0.38%p) 반영 후 손익비
  5. 평균 보유봉수 + 연환산근사(%) (시간효율)

생성되는 결과 파일: data/enha2_fibo_turn_backtest.csv
(엔하2+피보턴 최종 신호의 거래 1건 = 1행)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.enha2_fibo_turn import add_fibo_turn_signal, find_enha2_fibo_turn_events
from src.ma50_backtest import evaluate_trades, split_train_test
from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.reversal_angle import DEFAULT_N_BARS, add_all_indicators, simulate_reversal_trades
from src.triple_indicator_confluence import decide_confluence_window

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXISTING_COMPARISON_PATH = DATA_DIR / "reversal_angle_enhah_3_4_5_comparison.csv"
OUTPUT_PATH = DATA_DIR / "enha2_fibo_turn_backtest.csv"

# 2단계(거리 분포 확인)에서 넉넉하게 잡는 탐색 범위 - 직전 세션과 동일하게
# ±10거래일까지 보고, 실제 윈도우는 그 분포의 중앙값으로 자동 결정한다.
DISTANCE_SEARCH_WINDOW = 10

# 이전 세션(엔하5/엔하4 검증, run_enha_ensah_full_matrix.py)과 동일한 실전비용 가정.
BUY_FEE_PCT = 0.015
SELL_FEE_PCT = 0.165
SLIPPAGE_PCT_PER_SIDE = 0.1
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + 2 * SLIPPAGE_PCT_PER_SIDE

MIN_SAMPLE_FOR_CONCLUSION = 30

warnings = []


def collect_enha2_events():
    """
    9종목 순회하며 종목별 (df, 필터 전 엔하2+피보턴거리 이벤트표)를 모은다.
    지표 계산까지 한 번만 하고 df를 반환해서, 나중에 매매 시뮬레이션 때
    다시 계산하지 않도록 한다.
    """
    per_stock_df = {}
    per_stock_events = {}

    for label in KR_STOCK_CODES:
        print(f"[INFO] {label} 처리 중...")
        ticker, raw = fetch_kr_daily_bars(label)
        if raw.empty:
            warnings.append(f"[{label}] 데이터를 받아오지 못했습니다.")
            continue

        df = add_all_indicators(raw)
        df = add_fibo_turn_signal(df)

        events = find_enha2_fibo_turn_events(df, search_window=DISTANCE_SEARCH_WINDOW)
        if events.empty:
            warnings.append(f"[{label}] 엔하2 터치 이벤트가 하나도 없습니다.")
            continue

        events.insert(0, "티커", ticker)
        events.insert(0, "종목명", label)

        per_stock_df[label] = df
        per_stock_events[label] = events

    return per_stock_df, per_stock_events


def print_step1_sample_counts(all_events_raw, filtered_count, window):
    print("\n===== 1단계: 표본 수 (필터 전/후) =====")
    print(f"엔하2 터치 전체(필터 전): {len(all_events_raw)}건")
    print(f"피보턴이 ±{window}거래일 이내에 있는 경우(필터 후, 최종 신호): {filtered_count}건")


def build_enha2_fibo_trades(per_stock_df, per_stock_events, window):
    """윈도우로 걸러낸 최종 신호에 대해 종목별 매매 시뮬레이션을 실행하고 합친다."""
    all_trades = []

    for label, df in per_stock_df.items():
        events = per_stock_events[label]
        filtered = events[events["피보턴까지거리(일)"].abs() <= window].copy()
        if filtered.empty:
            continue

        trades = simulate_reversal_trades(df, filtered, n_bars=DEFAULT_N_BARS)
        trades = trades.merge(
            filtered[["터치시각", "피보턴까지거리(일)"]], on="터치시각", how="left"
        )

        train_part, _ = split_train_test(df, train_ratio=0.7)
        split_time = train_part.index[-1] if len(train_part) > 0 else df.index[-1]
        trades["구간"] = np.where(trades["터치시각"] <= split_time, "학습", "검증")

        trades.insert(0, "티커", filtered["티커"].iloc[0])
        trades.insert(0, "종목명", label)
        all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()

    combined = pd.concat(all_trades, ignore_index=True)
    return combined.sort_values(["종목명", "터치시각"]).reset_index(drop=True)


def summarize_signal(trades_df, label):
    """
    신호 하나(trades_df, "구간"/"손익률(%)"/"보유봉수" 컬럼 필요)에 대해
    학습/검증 구간별로 표본수/승률/손익비/비용반영손익비/평균보유봉수/
    연환산근사를 계산해서 요약 행 리스트(2행: 학습/검증)를 반환한다.
    """
    rows = []
    for 구간 in ["학습", "검증"]:
        g = trades_df[trades_df["구간"] == 구간]
        n = len(g)
        if n == 0:
            rows.append({"신호": label, "구간": 구간, "표본수": 0})
            continue

        m = evaluate_trades(g)

        cost_g = g.copy()
        cost_g["손익률(%)"] = cost_g["손익률(%)"] - ROUND_TRIP_COST_PCT
        m_cost = evaluate_trades(cost_g)

        avg_hold = g["보유봉수"].mean()
        avg_pnl = g["손익률(%)"].mean()
        annualized = (avg_pnl / avg_hold * 250) if avg_hold > 0 else np.nan

        rows.append(
            {
                "신호": label,
                "구간": 구간,
                "표본수": n,
                "승률(%)": m["승률(%)"],
                "손익비": m["손익비"],
                "손익비_비용반영": m_cost["손익비"],
                "평균보유봉수": avg_hold,
                "연환산근사(%)": annualized,
                "참고용_표본부족": n < MIN_SAMPLE_FOR_CONCLUSION,
            }
        )
    return rows


def load_existing_enha_3_4_5():
    """
    기존에 이미 검증해둔 엔하3/4/5 결과를 그대로 읽어온다 (재계산하지 않음).
    """
    if not EXISTING_COMPARISON_PATH.exists():
        warnings.append(
            f"{EXISTING_COMPARISON_PATH} 파일이 없어 엔하3/4/5 비교를 건너뜁니다. "
            "run_reversal_angle_enha_3_4_5_comparison.py를 먼저 실행해야 합니다."
        )
        return None
    return pd.read_csv(EXISTING_COMPARISON_PATH)


def print_comparison_table(summary_rows):
    print("\n===== 3~5단계: 엔하2+피보턴 vs 엔하3 vs 엔하4 vs 엔하5 비교표 =====")
    summary = pd.DataFrame(summary_rows)
    cols = ["신호", "구간", "표본수", "승률(%)", "손익비", "손익비_비용반영", "평균보유봉수", "연환산근사(%)"]
    print(summary[cols].round(2).to_string(index=False))

    print(
        f"\n(표본 {MIN_SAMPLE_FOR_CONCLUSION}건 미만인 구간은 참고용입니다. "
        "실전비용은 왕복 0.38%p(수수료+슬리피지, 이전 세션과 동일 가정)를 적용했습니다.)"
    )

    print(
        "\n[주의] 신호 4개(엔하2+피보턴, 엔하3, 엔하4, 엔하5)를 나란히 비교했으므로 "
        "다중비교 위험이 있습니다. '표본이 충분하면서(수백 건) 학습/검증 손익비가 "
        "둘 다 비용반영 후에도 1을 안정적으로 넘고 서로 크게 어긋나지 않는지'를 "
        "기준으로 판단해야지, 숫자만 보고 가장 높은 손익비 하나를 덥석 채택하면 안 됩니다."
    )


def main():
    per_stock_df, per_stock_events = collect_enha2_events()
    if not per_stock_events:
        print("[경고] 엔하2 터치 이벤트가 하나도 없습니다. 결과 파일을 만들지 않습니다.")
        return

    all_events_raw = pd.concat(per_stock_events.values(), ignore_index=True)

    # ---- 2단계: 거리 분포로 윈도우 자동 결정 (직전 세션과 동일 방법론) ----
    window = decide_confluence_window(list(all_events_raw["피보턴까지거리(일)"]))
    print(f"\n[INFO] 거리 분포를 바탕으로 정한 '근접' 판정 윈도우: ±{window}거래일")

    # ---- 최종 신호로 매매 시뮬레이션 ----
    enha2_trades = build_enha2_fibo_trades(per_stock_df, per_stock_events, window)
    print_step1_sample_counts(all_events_raw, len(enha2_trades), window)

    if enha2_trades.empty:
        print("[경고] 필터를 통과한 엔하2+피보턴 신호가 하나도 없습니다. 결과 파일을 만들지 않습니다.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    enha2_trades.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 엔하2+피보턴 최종 신호 {len(enha2_trades)}건을 저장했습니다: {OUTPUT_PATH}")

    # ---- 3~5단계: 기존 엔하3/4/5와 비교 ----
    summary_rows = summarize_signal(enha2_trades, "엔하2+피보턴")

    existing = load_existing_enha_3_4_5()
    if existing is not None:
        for 밴드 in ["엔하3", "엔하4", "엔하5"]:
            band_trades = existing[existing["밴드"] == 밴드]
            if band_trades.empty:
                continue
            summary_rows.extend(summarize_signal(band_trades, 밴드))

    print_comparison_table(summary_rows)

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
