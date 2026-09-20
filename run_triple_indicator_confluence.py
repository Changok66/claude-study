# -*- coding: utf-8 -*-
"""
"엔상하(50,20,1.5) + 선3일목(10,20,50) + 피보일목(5,8,13)" 세 지표가 같은 시점
근처에 동시에 신호를 낼 때 매매 효과(승률/손익비)가 더 좋아지는지 검증하는
실행 스크립트다. 지표/신호 계산 로직은 src/reversal_angle.py와
src/triple_indicator_confluence.py에 있고, 이 파일은 9종목을 순회하며 그
함수들을 순서대로 불러 쓰고 결과를 정리하는 역할만 한다.

검증 절차 (요청받은 4단계 그대로):
  1단계: 종목마다 신호1(엔상하)/신호2(선3일목)/신호3(피보일목) 발생일을 찾는다.
  2단계: 신호1 발생일 기준으로 신호2/신호3까지의 거리(거래일수) 분포를 본다.
  3단계: 그 분포로 정한 "동시성 윈도우"를 기준으로 1/2/3개 동시신호 그룹을
         나누고, 학습/검증 구간 x 그룹별 승률/손익비/보유기간을 비교한다.
  4단계: 그룹이 늘수록(1->2->3) 성과가 단조적으로 좋아지는지, 학습/검증
         구간에서 같은 경향이 재현되는지 확인한다.

생성되는 결과 파일: data/triple_indicator_confluence_backtest.csv
(거래 1건 = 1행. 학습/검증 구간, 동시신호개수 컬럼 포함)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.ma50_backtest import split_train_test
from src.reversal_angle import (
    DEFAULT_N_BARS,
    add_envelope_bands,
    add_fibo_5_8_13,
    add_span_10_20_50,
    simulate_reversal_trades,
)
from src.triple_indicator_confluence import (
    add_confluence_group,
    add_confluence_signals,
    compute_signal_distances,
    decide_confluence_window,
    find_signal1_events,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "triple_indicator_confluence_backtest.csv"

# 2단계에서 "신호2/신호3이 신호1과 관련 있다"고 볼 최대 범위(거래일).
# 이보다 멀리 떨어진 신호는 우연히 다른 시점에 뜬 것으로 보고 분포 계산에서
# 아예 제외한다(±10거래일 = 2주 정도면 "동시성"을 논하기에 충분히 넓은 범위).
DISTANCE_SEARCH_WINDOW = 10

# 표본이 이 개수 미만인 그룹은 "참고용"으로만 본다 (요청받은 주의사항).
MIN_SAMPLE_SIZE = 30

warnings = []


def _win_rate(pnl_series):
    """손익률(%) 목록을 받아서 승률(%)을 계산한다. 표본이 없으면 None."""
    if len(pnl_series) == 0:
        return None
    return (pnl_series > 0).mean() * 100


def _profit_factor(pnl_series):
    """
    손익비 = 평균 이익 / |평균 손실|.
    이긴 거래가 하나도 없거나 진 거래가 하나도 없으면 비교 자체가 무의미하므로
    None을 반환한다.
    """
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    if len(wins) == 0 or len(losses) == 0:
        return None
    return wins.mean() / abs(losses.mean())


def process_one_stock(label, ticker):
    """
    종목 1개를 받아서:
      - 지표/신호 계산까지 마친 일봉(df)
      - 신호1(엔상하) 이벤트에 신호2/3까지 거리까지 붙인 표(events)
    를 튜플로 반환한다. 데이터가 없거나 신호1 이벤트가 하나도 없으면
    (None, None)을 반환한다.
    """
    ticker_actual, df = fetch_kr_daily_bars(label)
    if df.empty:
        warnings.append(f"[{label}] 데이터를 받아오지 못했습니다 (야후 파이낸스 조회 실패).")
        return None, None

    # ---- 지표 계산 (전체 기간을 한 번에 계산해야 MA/ATR 워밍업이 검증구간
    #      앞부분에서 끊기지 않는다 - 기존 run_reversal_angle_analysis.py와 동일) ----
    df = add_envelope_bands(df)
    df = add_span_10_20_50(df)
    df = add_fibo_5_8_13(df)
    df = add_confluence_signals(df)

    # ---- 1단계: 신호1(엔상하) 이벤트 목록화 ----
    events1 = find_signal1_events(df)
    if events1.empty:
        warnings.append(f"[{label}] 신호1(엔상하 엔하4/5 터치) 이벤트가 하나도 발견되지 않았습니다.")
        return None, None

    # ---- 2단계 준비: 신호1 각 이벤트에서 신호2/신호3까지 거리 계산 ----
    events1 = compute_signal_distances(df, events1, search_window=DISTANCE_SEARCH_WINDOW)
    events1.insert(0, "티커", ticker_actual)
    events1.insert(0, "종목명", label)

    return df, events1


def print_step2_distance_summary(all_events):
    """2단계: 신호1 기준 신호2/신호3까지 거리 분포를 콘솔에 출력한다."""
    print("\n===== 2단계: 신호 간 시간 거리 분포 (9종목 신호1 이벤트 전체 합산) =====")
    for col, name in [("신호2까지거리(일)", "신호2(선3일목)"), ("신호3까지거리(일)", "신호3(피보일목)")]:
        values = all_events[col].dropna()
        found_ratio = len(values) / len(all_events) * 100 if len(all_events) > 0 else 0
        print(f"[{name}] 발견율 {found_ratio:.1f}% ({len(values)}/{len(all_events)}건, "
              f"±{DISTANCE_SEARCH_WINDOW}거래일 이내 기준)")
        if len(values) > 0:
            print(
                f"  거리(일) 분포: 평균 {values.mean():.2f}, 중앙값 {values.median():.2f}, "
                f"25%~75% 구간 [{values.quantile(0.25):.1f}, {values.quantile(0.75):.1f}]"
            )


def print_step3_group_summary(trades):
    """3단계: 구간 x 동시신호개수 그룹별 승률/손익비/평균보유봉수를 출력한다."""
    print("\n===== 3단계: 구간 x 동시신호개수 그룹별 성과 =====")
    grouped = trades.groupby(["구간", "동시신호개수"])

    for (구간, 그룹), g in grouped:
        pnl = g["손익률(%)"]
        win_rate = _win_rate(pnl)
        profit_factor = _profit_factor(pnl)
        avg_hold = g["보유봉수"].mean()

        flag = " (참고용, n<30)" if len(g) < MIN_SAMPLE_SIZE else ""
        pf_text = f"{profit_factor:.2f}" if profit_factor is not None else "계산불가"
        print(
            f"[{구간}] 동시신호 {그룹}개: {len(g)}건{flag} - "
            f"승률 {win_rate:.1f}%, 손익비 {pf_text}, 평균보유 {avg_hold:.1f}봉"
        )


def check_monotonic(trades, 구간):
    """
    4단계: 지정한 구간(학습/검증) 안에서 동시신호개수 1->2->3으로 갈수록
    승률과 손익비가 단조 증가하는지 확인해서 (증가여부, 그룹별 요약표)를
    반환한다.
    """
    subset = trades[trades["구간"] == 구간]
    rows = []
    for group in sorted(subset["동시신호개수"].unique()):
        g = subset[subset["동시신호개수"] == group]
        rows.append(
            {
                "동시신호개수": group,
                "건수": len(g),
                "승률(%)": _win_rate(g["손익률(%)"]),
                "손익비": _profit_factor(g["손익률(%)"]),
            }
        )
    summary = pd.DataFrame(rows).sort_values("동시신호개수").reset_index(drop=True)

    # 승률/손익비 둘 다 결측 없이 이전 그룹보다 커지는 경우만 "단조 증가"로 인정.
    def _is_monotonic(series):
        values = series.dropna().tolist()
        if len(values) < 2:
            return None  # 비교할 그룹이 2개 미만이면 판단 불가
        return all(values[i] < values[i + 1] for i in range(len(values) - 1))

    win_rate_monotonic = _is_monotonic(summary["승률(%)"])
    profit_factor_monotonic = _is_monotonic(summary["손익비"])

    return summary, win_rate_monotonic, profit_factor_monotonic


def print_step4_monotonic_check(trades):
    print("\n===== 4단계: 동시신호개수가 늘수록 성과가 단조적으로 좋아지는가 =====")
    for 구간 in ["학습", "검증"]:
        summary, win_ok, pf_ok = check_monotonic(trades, 구간)
        print(f"\n[{구간} 구간]")
        print(summary.to_string(index=False))

        def _fmt(ok):
            if ok is None:
                return "판단불가(비교할 그룹 부족)"
            return "예 (단조 증가)" if ok else "아니오"

        print(f"  -> 승률 단조 증가 여부: {_fmt(win_ok)}")
        print(f"  -> 손익비 단조 증가 여부: {_fmt(pf_ok)}")

    print(
        "\n[주의] 동시신호개수 1/2/3, 이렇게 3개 그룹으로 나눠 비교했으므로 "
        "그중 하나는 우연히 좋아 보일 수 있습니다(다중비교 문제). 학습구간과 "
        "검증구간 양쪽에서 같은 경향(단조 증가)이 재현될 때만 신뢰할 수 있는 "
        "패턴으로 봐야 합니다."
    )


def print_sanity_check(trades):
    """그룹별 건수 합이 전체 건수와 일치하는지 손으로 검산하듯 확인한다."""
    total = len(trades)
    group_sum = trades["동시신호개수"].value_counts().sum()
    match = "일치" if total == group_sum else "불일치!! 확인 필요"
    print(f"\n[검산] 전체 거래 {total}건 vs 그룹별 합계 {group_sum}건 -> {match}")


def main():
    per_stock_df = {}
    per_stock_events = {}

    # ---- 1단계 + 2단계 재료 모으기: 9종목 순회 ----
    for label in KR_STOCK_CODES:
        print(f"[INFO] {label} 처리 중...")
        df, events1 = process_one_stock(label, label)
        if df is None:
            continue
        per_stock_df[label] = df
        per_stock_events[label] = events1

    if not per_stock_events:
        print("[경고] 9종목 전부에서 신호1 이벤트가 발견되지 않았습니다. 결과 파일을 만들지 않습니다.")
        return

    all_events_raw = pd.concat(per_stock_events.values(), ignore_index=True)
    print_step2_distance_summary(all_events_raw)

    # ---- 2단계 결과로 3단계에서 쓸 동시성 윈도우 결정 ----
    all_distances = list(all_events_raw["신호2까지거리(일)"]) + list(all_events_raw["신호3까지거리(일)"])
    window = decide_confluence_window(all_distances)
    print(f"\n[INFO] 2단계 분포를 바탕으로 정한 동시성 판정 윈도우: ±{window}거래일")

    # ---- 3단계: 동시신호개수 그룹 부여 + 매매 시뮬레이션 ----
    all_trades = []
    for label, df in per_stock_df.items():
        events1 = per_stock_events[label]
        events1 = add_confluence_group(events1, window)

        trades = simulate_reversal_trades(df, events1, n_bars=DEFAULT_N_BARS)
        if trades.empty:
            continue

        # simulate_reversal_trades는 "터치시각" 기준으로 events1과 같은 순서의
        # 행을 만들어주므로, 부가 정보(동시신호개수 등)를 터치시각으로 합친다.
        extra_cols = events1[
            ["터치시각", "동시신호개수", "신호2존재여부", "신호2까지거리(일)",
             "신호3존재여부", "신호3까지거리(일)"]
        ]
        trades = trades.merge(extra_cols, on="터치시각", how="left")

        # 학습(70%)/검증(30%) 구간 라벨 - 기존 run_reversal_angle_analysis.py와 동일한 방식.
        train_part, _ = split_train_test(df, train_ratio=0.7)
        split_time = train_part.index[-1] if len(train_part) > 0 else df.index[-1]
        trades["구간"] = np.where(trades["터치시각"] <= split_time, "학습", "검증")

        trades.insert(0, "티커", events1["티커"].iloc[0])
        trades.insert(0, "종목명", label)

        all_trades.append(trades)

    if not all_trades:
        print("[경고] 매매 시뮬레이션 결과가 하나도 없습니다. 결과 파일을 만들지 않습니다.")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values(["종목명", "터치시각"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 총 {len(combined)}건의 거래를 저장했습니다: {OUTPUT_PATH}")

    print_step3_group_summary(combined)
    print_step4_monotonic_check(combined)
    print_sanity_check(combined)

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
