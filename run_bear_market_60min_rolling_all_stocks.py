# -*- coding: utf-8 -*-
"""
하락장 60분봉 롤링매매 전략(src/bear_market_60min_rolling.py)을 9종목
전체로 확장해서 표본을 늘리는 실행 스크립트다. 삼성전자 1종목(13건)만으로는
표본이 부족하다는 요청에 따라, 같은 국면판정 로직(일봉 이격도 10일
이동평균이 20거래일 이상 연속 음수)으로 9종목 각각의 "진짜 하락구간"을
찾고, 그 구간에 같은 롤링매매 전략을 적용한다.

핵심 제약: 60분봉 데이터(data/price_{code}_min60.csv)는 9종목 전부
2025-09-01~2026-09-04(약 1년)만 있다. 국면판정은 일봉 전체 역사로 하지만,
실제 백테스트는 60분봉이 있는 이 1년 구간과 겹치는 부분만 가능하다 - 그래서
각 종목의 하락구간을 60분봉 보유기간과 교집합(clip)한 뒤에만 백테스트한다.
교집합이 없거나 너무 짧으면(20봉 미만) 그 구간은 건너뛴다.

생성되는 결과 파일: data/bear_market_60min_rolling_all_stocks.csv
(라운드 1건 = 1행, 종목명/하락구간 컬럼 포함)
"""

from pathlib import Path

import pandas as pd

from src.bear_market_60min_rolling import (
    add_buy_sell_signals,
    find_bear_market_periods,
    simulate_rolling_trades,
)
from src.ma50_backtest import evaluate_trades
from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars, fetch_kr_min60_bars
from src.reversal_angle import add_all_indicators

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "bear_market_60min_rolling_all_stocks.csv"

MIN_BARS_FOR_BACKTEST = 20  # 교집합 구간이 이보다 짧으면 의미가 없다고 보고 건너뜀

BUY_FEE_PCT = 0.015
SELL_FEE_PCT = 0.165
SLIPPAGE_PCT_PER_SIDE = 0.1
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + 2 * SLIPPAGE_PCT_PER_SIDE

warnings = []


def clip_periods_to_min60_range(periods, min60_start, min60_end):
    """일봉 기준 하락구간 목록을 60분봉 보유기간과 교집합해서 자른다."""
    clipped = []
    for start, end in periods:
        c_start = max(start.tz_localize(None), min60_start)
        c_end = min(end.tz_localize(None), min60_end)
        if c_start <= c_end:
            clipped.append((c_start, c_end))
    return clipped


def process_one_stock(label):
    """
    종목 1개를 받아서, 하락구간(60분봉 보유기간과 교집합)마다 롤링매매를
    실행한 거래 표를 반환한다. 구간이 하나도 없으면 빈 데이터프레임.
    """
    ticker, daily_raw = fetch_kr_daily_bars(label)
    if daily_raw.empty:
        warnings.append(f"[{label}] 일봉 데이터를 받아오지 못했습니다.")
        return pd.DataFrame()

    code, min60_raw = fetch_kr_min60_bars(label)
    if min60_raw.empty:
        warnings.append(f"[{label}] 60분봉 데이터가 없습니다.")
        return pd.DataFrame()

    bear_periods = find_bear_market_periods(daily_raw)
    clipped = clip_periods_to_min60_range(bear_periods, min60_raw.index.min(), min60_raw.index.max())
    if not clipped:
        warnings.append(f"[{label}] 60분봉 보유기간과 겹치는 하락구간이 없습니다.")
        return pd.DataFrame()

    # 지표는 60분봉 전체 기간으로 먼저 계산해서 워밍업이 백테스트 구간
    # 앞부분에서 끊기지 않게 한다.
    full_df = add_all_indicators(min60_raw)

    all_trades = []
    for period_no, (c_start, c_end) in enumerate(clipped, start=1):
        window_df = full_df.loc[c_start:c_end].copy()
        if len(window_df) < MIN_BARS_FOR_BACKTEST:
            warnings.append(
                f"[{label}] 하락구간#{period_no}({c_start.date()}~{c_end.date()})이 "
                f"{len(window_df)}봉뿐이라 건너뜁니다(최소 {MIN_BARS_FOR_BACKTEST}봉 필요)."
            )
            continue

        window_df = add_buy_sell_signals(window_df)
        trades = simulate_rolling_trades(window_df)

        buy_hold_pct = (window_df["Close"].iloc[-1] / window_df["Close"].iloc[0] - 1) * 100
        print(
            f"  [{label}] 하락구간#{period_no} {c_start.date()}~{c_end.date()} "
            f"({len(window_df)}봉): 단순보유 {buy_hold_pct:+.2f}%, 라운드 {len(trades)}건"
        )

        if trades.empty:
            continue

        trades.insert(0, "하락구간번호", period_no)
        trades.insert(0, "하락구간시작", c_start.date())
        trades.insert(0, "하락구간_단순보유수익률(%)", round(buy_hold_pct, 2))
        trades.insert(0, "티커", code)
        trades.insert(0, "종목명", label)
        all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True)


def print_overall_summary(combined):
    print("\n===== 전체 종합 (9종목 하락구간 롤링매매 합산) =====")
    m = evaluate_trades(combined)
    cost = combined.copy()
    cost["손익률(%)"] = cost["손익률(%)"] - ROUND_TRIP_COST_PCT
    m_cost = evaluate_trades(cost)
    print(f"  총 라운드 수: {m['거래수']}건")
    print(f"  승률: {m['승률(%)']:.1f}%")
    print(f"  손익비: {m['손익비']:.2f} (비용반영: {m_cost['손익비']:.2f})")
    print(f"  평균 보유봉수: {combined['보유봉수'].mean():.1f}봉")

    print("\n  종목별 라운드 수/승률/손익비:")
    for label, g in combined.groupby("종목명"):
        gm = evaluate_trades(g)
        print(f"    {label}: {gm['거래수']}건, 승률 {gm['승률(%)']:.1f}%, 손익비 {gm['손익비']:.2f}")

    print("\n  하락구간별 단순보유 대비:")
    period_view = combined.drop_duplicates(subset=["종목명", "하락구간번호"])[
        ["종목명", "하락구간시작", "하락구간번호", "하락구간_단순보유수익률(%)"]
    ]
    print(period_view.to_string(index=False))


def main():
    all_trades = []
    for label in KR_STOCK_CODES:
        print(f"[INFO] {label} 처리 중...")
        trades = process_one_stock(label)
        if not trades.empty:
            all_trades.append(trades)

    if not all_trades:
        print("[경고] 9종목 전부에서 라운드가 하나도 없습니다.")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values(["종목명", "하락구간번호", "진입일시"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 총 {len(combined)}개 라운드를 저장했습니다: {OUTPUT_PATH}")

    print_overall_summary(combined)

    if warnings:
        print("\n===== 경고/참고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
