# -*- coding: utf-8 -*-
"""
엔하2+피보턴을 주신호로 유지하고, 새 추세선(수식4=MA13+MA19평균,
수식5=MA15+MA21평균)의 방향을 "참조용 필터/등급"으로만 붙여서 효과를
검증하는 실행 스크립트다.

방법A(필터): 추세선 방향이 "상승"인 신호만 채택, "하락"이면 스킵
방법B(등급): 거래는 버리지 않고 "강한신호"(방향 일치)/"약한신호"(불일치)로
             등급만 나눠서 승률/손익비 차이를 비교

학습/검증 분리는 2026-09-18 세션에서 발견한 "종목별 자기 역사 뒤 30%" 방식의
상승장 편향 문제(세션정리_학습검증분리방법론한계_20260918.md)를 피하기 위해,
9종목 중 2020년 이전부터 데이터가 있는 7종목에 공통 달력 기간(학습
2020-01-01~2022-12-31, 검증 2023-01-01~)을 적용한다. HD현대중공업(2021-09
상장)과 에이피알(2024-02 상장)은 이 공통기간에 낄 수 없어 제외한다(결과
CSV에는 포함하되 "공통기간제외여부" 플래그로 표시).

생성되는 결과 파일: data/enha2_fibo_trendline_filter_backtest.csv
(9종목 엔하2+피보턴 거래 전체, 추세선방향/강약등급/공통기간구간 컬럼 포함)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.enha2_fibo_trendline_filter import attach_trendline_direction
from src.enha2_fibo_turn import add_fibo_turn_signal, find_enha2_fibo_turn_events
from src.fibo_trendline_breakout import add_trend_lines
from src.ma50_backtest import evaluate_trades, split_by_common_period
from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.reversal_angle import DEFAULT_N_BARS, add_all_indicators, simulate_reversal_trades
from src.triple_indicator_confluence import decide_confluence_window

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "enha2_fibo_trendline_filter_backtest.csv"
EXISTING_COMPARISON_PATH = DATA_DIR / "enha2_fibo_turn_backtest.csv"

DISTANCE_SEARCH_WINDOW = 10
MIN_SAMPLE_FOR_CONCLUSION = 30

# 이전 세션(엔하2+피보턴 검증)과 동일한 실전비용 가정.
BUY_FEE_PCT = 0.015
SELL_FEE_PCT = 0.165
SLIPPAGE_PCT_PER_SIDE = 0.1
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + 2 * SLIPPAGE_PCT_PER_SIDE

# 2017년 이전부터 데이터가 있어 "2020년 학습 시작"을 온전히 커버하는 7종목만
# 공통기간 비교에 포함한다 (HD현대중공업 2021-09, 에이피알 2024-02 상장이라 제외).
COMMON_PERIOD_COHORT = [
    "삼성전자", "SK하이닉스", "두산에너빌리티", "테크윙", "아모레퍼시픽",
    "HD현대일렉트릭", "대한광통신",
]
TRAIN_START, TRAIN_END = "2020-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2030-01-01"  # 종료일은 사실상 "데이터 최신일까지" 열어둠

warnings = []


def process_one_stock(label):
    """
    종목 1개를 받아서 (df, events1)을 반환한다.
    df: add_all_indicators + add_fibo_turn_signal + add_trend_lines를 거친 일봉
    events1: 엔하2 터치 이벤트 + 피보턴까지거리(일) 컬럼
    """
    ticker, raw = fetch_kr_daily_bars(label)
    if raw.empty:
        warnings.append(f"[{label}] 데이터를 받아오지 못했습니다.")
        return None, None, None

    df = add_all_indicators(raw)
    df = add_fibo_turn_signal(df)
    df = add_trend_lines(df)

    events1 = find_enha2_fibo_turn_events(df, search_window=DISTANCE_SEARCH_WINDOW)
    if events1.empty:
        warnings.append(f"[{label}] 엔하2 터치 이벤트가 없습니다.")
        return None, None, None

    events1.insert(0, "티커", ticker)
    events1.insert(0, "종목명", label)
    return ticker, df, events1


def build_trades(df, events1, window, ticker, label):
    """윈도우로 걸러낸 엔하2+피보턴 신호에 대해 매매 시뮬레이션 + 추세선 방향/공통기간 라벨 부착."""
    filtered = events1[events1["피보턴까지거리(일)"].abs() <= window].copy()
    if filtered.empty:
        return pd.DataFrame()

    trades = simulate_reversal_trades(df, filtered, n_bars=DEFAULT_N_BARS)
    trades = trades.merge(filtered[["터치시각"]], on="터치시각", how="left")
    trades = attach_trendline_direction(df, trades)

    common_labels = split_by_common_period(df, TRAIN_START, TRAIN_END, VAL_START, VAL_END)
    date_to_label = {ts.date(): lab for ts, lab in common_labels.items()}
    trades["공통기간구간"] = trades["터치시각"].apply(lambda ts: date_to_label.get(ts.date()))
    trades["공통기간제외여부"] = label not in COMMON_PERIOD_COHORT

    trades.insert(0, "티커", ticker)
    trades.insert(0, "종목명", label)
    return trades


def summarize(trades_df, label):
    """구간(학습/검증)별 표본수/승률/손익비/비용반영손익비/평균보유봉수.
    trades_df는 "구간"(기존 per-종목 분리) 또는 "공통기간구간"(이번 신규
    분리) 둘 중 하나를 갖고 있다 - 있는 쪽을 그대로 쓴다."""
    period_col = "구간" if "구간" in trades_df.columns else "공통기간구간"
    rows = []
    for 구간 in ["학습", "검증"]:
        g = trades_df[trades_df[period_col] == 구간]
        n = len(g)
        if n == 0:
            rows.append({"신호": label, "구간": 구간, "표본수": 0})
            continue

        m = evaluate_trades(g)
        cost_g = g.copy()
        cost_g["손익률(%)"] = cost_g["손익률(%)"] - ROUND_TRIP_COST_PCT
        m_cost = evaluate_trades(cost_g)
        avg_hold = g["보유봉수"].mean()

        rows.append({
            "신호": label, "구간": 구간, "표본수": n,
            "승률(%)": m["승률(%)"], "손익비": m["손익비"],
            "손익비_비용반영": m_cost["손익비"], "평균보유봉수": avg_hold,
        })
    return rows


def print_table(rows, title):
    print(f"\n===== {title} =====")
    df = pd.DataFrame(rows)
    cols = ["신호", "구간", "표본수", "승률(%)", "손익비", "손익비_비용반영", "평균보유봉수"]
    print(df[cols].round(2).to_string(index=False))


def print_common_period_return(per_stock_df):
    print("\n===== 참고: 공통 검증기간(2023~) 7종목 수익률 (상승장 여부 확인용) =====")
    for label in COMMON_PERIOD_COHORT:
        df = per_stock_df.get(label)
        if df is None:
            continue
        val_slice = df.loc[VAL_START:]
        if val_slice.empty:
            continue
        change_pct = (val_slice["Close"].iloc[-1] / val_slice["Close"].iloc[0] - 1) * 100
        print(f"  {label}: {val_slice.index[0].date()}~{val_slice.index[-1].date()} 수익률 {change_pct:+.1f}%")


def main():
    per_stock_df = {}
    per_stock_events = {}

    for label in KR_STOCK_CODES:
        print(f"[INFO] {label} 처리 중...")
        ticker, df, events1 = process_one_stock(label)
        if df is None:
            continue
        per_stock_df[label] = df
        per_stock_events[label] = (ticker, events1)

    if not per_stock_events:
        print("[경고] 처리된 종목이 없습니다.")
        return

    all_events_raw = pd.concat([ev for _, ev in per_stock_events.values()], ignore_index=True)
    window = decide_confluence_window(list(all_events_raw["피보턴까지거리(일)"]))
    print(f"\n[INFO] 동시성 판정 윈도우: ±{window}거래일")

    all_trades = []
    for label, df in per_stock_df.items():
        ticker, events1 = per_stock_events[label]
        trades = build_trades(df, events1, window, ticker, label)
        if trades.empty:
            continue
        all_trades.append(trades)

    if not all_trades:
        print("[경고] 매매 시뮬레이션 결과가 없습니다.")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values(["종목명", "터치시각"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 총 {len(combined)}건을 저장했습니다: {OUTPUT_PATH}")

    # 공통기간 비교 대상(7종목, 공통기간 라벨이 학습/검증인 거래만)으로 좁힌다.
    cohort = combined[
        (~combined["공통기간제외여부"]) & combined["공통기간구간"].isin(["학습", "검증"])
    ].copy()
    print(f"[검산] 공통기간 비교 대상(7종목, 학습+검증): {len(cohort)}건 "
          f"(9종목 전체 {len(combined)}건 중 제외/공백기간 {len(combined) - len(cohort)}건)")

    # ---- 기준선(필터 없음, 공통기간 재계산) ----
    baseline_rows = []
    for 구간 in ["학습", "검증"]:
        g = cohort[cohort["공통기간구간"] == 구간]
        n = len(g)
        if n == 0:
            baseline_rows.append({"신호": "기준선(필터없음,공통기간)", "구간": 구간, "표본수": 0})
            continue
        m = evaluate_trades(g)
        cost_g = g.copy()
        cost_g["손익률(%)"] = cost_g["손익률(%)"] - ROUND_TRIP_COST_PCT
        m_cost = evaluate_trades(cost_g)
        baseline_rows.append({
            "신호": "기준선(필터없음,공통기간)", "구간": 구간, "표본수": n,
            "승률(%)": m["승률(%)"], "손익비": m["손익비"],
            "손익비_비용반영": m_cost["손익비"], "평균보유봉수": g["보유봉수"].mean(),
        })
    print_table(baseline_rows, "기준선 - 필터 없음 (공통기간 재계산, 7종목)")

    # ---- 방법A: 필터 ----
    for 수식 in ["수식4", "수식5"]:
        col = f"추세선방향({수식})"
        filtered = cohort[cohort[col] == "상승"]
        rows = []
        for 구간 in ["학습", "검증"]:
            g = filtered[filtered["공통기간구간"] == 구간]
            g_all = cohort[cohort["공통기간구간"] == 구간]
            n, n_all = len(g), len(g_all)
            if n == 0:
                rows.append({"신호": f"방법A-{수식}", "구간": 구간, "표본수": 0})
                continue
            m = evaluate_trades(g)
            cost_g = g.copy()
            cost_g["손익률(%)"] = cost_g["손익률(%)"] - ROUND_TRIP_COST_PCT
            m_cost = evaluate_trades(cost_g)
            rows.append({
                "신호": f"방법A-{수식}", "구간": 구간, "표본수": n,
                "승률(%)": m["승률(%)"], "손익비": m["손익비"],
                "손익비_비용반영": m_cost["손익비"], "평균보유봉수": g["보유봉수"].mean(),
            })
            print(f"  [{수식}][{구간}] 필터 전 {n_all}건 -> 필터 후 {n}건 "
                  f"({n / n_all * 100:.1f}% 유지)")
        print_table(rows, f"방법A - 추세선 상승일 때만 채택 ({수식})")

    # ---- 방법B: 등급 ----
    for 수식 in ["수식4", "수식5"]:
        col = f"강약등급({수식})"
        rows = []
        for 등급 in ["강한신호", "약한신호"]:
            g_grade = cohort[cohort[col] == 등급]
            rows.extend(summarize(g_grade, f"{등급}({수식})"))
        print_table(rows, f"방법B - 강한신호 vs 약한신호 ({수식})")

    # ---- 기존 608건(per-종목 70/30 분리)과 비교 ----
    if EXISTING_COMPARISON_PATH.exists():
        existing = pd.read_csv(EXISTING_COMPARISON_PATH, encoding="utf-8-sig")
        existing_rows = summarize(existing, "엔하2+피보턴(기존, per종목 70/30)")
        print_table(existing_rows, "참고 - 기존 608건 (per-종목 분리 방식, 방법론 다름 주의)")
    else:
        warnings.append(f"{EXISTING_COMPARISON_PATH} 없음 - 기존 결과 비교 생략")

    print_common_period_return(per_stock_df)

    print(
        "\n[주의] 방법A(2개) + 방법B(강/약 x 2개) + 기준선 + 기존 결과까지 "
        "여러 개를 나란히 비교했으므로 다중비교 위험이 있습니다. 또한 기존 608건은 "
        "종목별 70/30 분리, 이번 결과는 7종목 공통기간 분리로 방법론 자체가 달라서 "
        "직접 비교 시 그 차이도 함께 감안해야 합니다."
    )

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
