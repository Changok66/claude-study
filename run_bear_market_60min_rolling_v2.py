# -*- coding: utf-8 -*-
"""
하락장 60분봉 롤링매매 - 국면판정 v2 (실제 가격 등락률 기준으로 재검증).

run_bear_market_60min_rolling_all_stocks.py에서 쓴 국면판정(이격도 10일
이동평균이 20거래일 이상 연속 음수)은 "가격이 MA50보다 낮다"만 볼 뿐 "그
구간 동안 가격이 실제로 떨어졌다"는 보장하지 않아서, HD현대중공업 등 여러
후보 구간이 실제로는 상승 구간이었던 문제가 발견됐다(9종목 확장 검증 중).

이 스크립트는 같은 후보 구간(이격도 기반, find_bear_market_periods)을 60분봉
보유기간과 교집합(clip)한 뒤, **그 잘린(=실제로 백테스트할) 구간의 시작가
대비 종가가 -5% 이상 하락했는지**를 추가로 확인해서 "진짜 하락구간"만
남긴다(src.bear_market_60min_rolling.is_true_decline). 처음엔 자르기 전
전체 일봉 후보 구간으로 등락률을 판정했었는데, 그러면 교집합으로 잘려나간
부분의 등락까지 섞여서 실제 백테스트 구간과 다른(심지어 반대) 결과가
나오는 버그가 있어서(삼성전자 -7% 구간이 빠지고, 아모레퍼시픽 +1% 구간이
남는 역설 발생) 잘린 뒤의 구간으로 판정하도록 고쳤다.

생성되는 결과 파일: data/bear_market_60min_rolling_v2_true_decline.csv
(라운드 1건 = 1행)
"""

from pathlib import Path

import pandas as pd

from src.bear_market_60min_rolling import (
    add_buy_sell_signals,
    find_bear_market_periods,
    is_true_decline,
    simulate_rolling_trades,
)
from src.ma50_backtest import evaluate_trades
from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars, fetch_kr_min60_bars
from src.reversal_angle import add_all_indicators

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "bear_market_60min_rolling_v2_true_decline.csv"

MIN_BARS_FOR_BACKTEST = 20
MIN_DECLINE_PCT = -5.0  # "진짜 하락구간" 판정 기준 (예시 그대로: -5% 이하)

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
    """종목 1개를 받아서, '진짜 하락구간'(60분봉 보유기간과 교집합)마다 롤링매매를 실행한다."""
    ticker, daily_raw = fetch_kr_daily_bars(label)
    if daily_raw.empty:
        warnings.append(f"[{label}] 일봉 데이터를 받아오지 못했습니다.")
        return pd.DataFrame()

    code, min60_raw = fetch_kr_min60_bars(label)
    if min60_raw.empty:
        warnings.append(f"[{label}] 60분봉 데이터가 없습니다.")
        return pd.DataFrame()

    candidates = find_bear_market_periods(daily_raw)
    clipped = clip_periods_to_min60_range(candidates, min60_raw.index.min(), min60_raw.index.max())
    if not clipped:
        warnings.append(f"[{label}] 60분봉 보유기간과 겹치는 후보 하락구간이 없습니다.")
        return pd.DataFrame()

    full_df = add_all_indicators(min60_raw)

    all_trades = []
    for period_no, (c_start, c_end) in enumerate(clipped, start=1):
        window_df = full_df.loc[c_start:c_end].copy()
        if len(window_df) < MIN_BARS_FOR_BACKTEST:
            warnings.append(
                f"[{label}] 후보구간#{period_no}({c_start.date()}~{c_end.date()})이 "
                f"{len(window_df)}봉뿐이라 건너뜁니다(최소 {MIN_BARS_FOR_BACKTEST}봉 필요)."
            )
            continue

        # 잘린(=실제로 백테스트할) 구간 자체의 등락률로 "진짜 하락구간"인지 확인.
        # (자르기 전 전체 후보구간으로 판정하면 안 됨 - docstring 참고)
        passed, buy_hold_pct = is_true_decline(window_df["Close"], min_decline_pct=MIN_DECLINE_PCT)
        if not passed:
            print(
                f"  [{label}] 후보구간#{period_no} {c_start.date()}~{c_end.date()} "
                f"({len(window_df)}봉): 실제등락률 {buy_hold_pct:+.2f}%로 "
                f"{MIN_DECLINE_PCT:.0f}% 기준 미달 - 제외"
            )
            continue

        window_df = add_buy_sell_signals(window_df)
        trades = simulate_rolling_trades(window_df)

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
    print("\n===== 전체 종합 (9종목, v2 진짜하락구간 롤링매매 합산) =====")
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

    print("\n  하락구간별 단순보유 대비 (전부 -5% 이하여야 정상):")
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
        print("[경고] 9종목 전부에서 '진짜 하락구간' 라운드가 하나도 없습니다.")
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
