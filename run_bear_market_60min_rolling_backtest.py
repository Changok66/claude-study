# -*- coding: utf-8 -*-
"""
하락장 국면에서 60분봉으로 "저점 매수(피보턴 또는 엔하4/5 반전턴) - MA50 근처
매도(엔상1 터치+하방턴)"를 반복하는 롤링매매를 검증하는 실행 스크립트다.

1차 실행(기본값, 인자 없이 실행): 삼성전자, 사용자가 처음 지정한
2026-03-01~2026-09-18 구간의 60분봉(data/price_005930_min60.csv)을 대상으로
한다. 그런데 이 구간은 실행 후 확인해보니 단순보유로도 +20.5%인 상승 포함
구간이었다(일봉 이격도 기준 진짜 하락 지속 구간은 2026-07-13~09-04로 더
좁게 나옴). 그래서 --start/--end/--output으로 다른 기간을 지정해 재실행할
수 있게 만들어서, "상승장 포함 원래 구간"과 "진짜 하락구간"을 각각 별도
파일로 저장해 비교할 수 있게 했다. 9종목 확장은 이번 범위에 포함하지
않는다(다음 단계 후보).

기본 결과 파일: data/bear_market_60min_rolling_backtest.csv (라운드 1건=1행)
"""

import argparse
from pathlib import Path

import pandas as pd

from src.bear_market_60min_rolling import (
    add_buy_sell_signals,
    find_bear_market_periods,
    simulate_rolling_trades,
)
from src.ma50_backtest import evaluate_trades
from src.ma50_data import fetch_kr_daily_bars, fetch_kr_min60_bars
from src.reversal_angle import add_all_indicators

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_PATH = DATA_DIR / "bear_market_60min_rolling_backtest.csv"

LABEL = "삼성전자"
DEFAULT_WINDOW_START = "2026-03-01"
DEFAULT_WINDOW_END = "2026-09-18"

# 이전 세션들과 동일한 실전비용 가정(왕복 0.38%p).
BUY_FEE_PCT = 0.015
SELL_FEE_PCT = 0.165
SLIPPAGE_PCT_PER_SIDE = 0.1
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + 2 * SLIPPAGE_PCT_PER_SIDE


def print_phase_check(window_start, window_end):
    """참고용: 지정한 구간이 실제로 일봉 이격도 기준 하락국면과 겹치는지 확인."""
    print("\n===== 참고: 일봉 이격도 기준 국면 확인 =====")
    _, daily_raw = fetch_kr_daily_bars(LABEL)
    periods = find_bear_market_periods(daily_raw)
    overlapped = [
        (s, e) for s, e in periods
        if not (e < pd.Timestamp(window_start, tz=daily_raw.index.tz)
                or s > pd.Timestamp(window_end, tz=daily_raw.index.tz))
    ]
    if overlapped:
        print(f"  이격도(10일 이동평균) 기준 하락 국면(20거래일 이상 지속)이 지정 구간과 겹칩니다:")
        for s, e in overlapped:
            print(f"    {s.date()} ~ {e.date()}")
    else:
        print("  이 기준으로는 지정 구간과 겹치는 '20거래일 이상 지속 하락' 구간을 못 찾았습니다 "
              "(그래도 사용자가 직접 확인한 구간이므로 그대로 진행합니다).")


def print_signal_frequency(df):
    print("\n===== 매수 신호 빈도 (조건별, 중복 포함 - 같은 봉에 여러 조건 동시 성립 가능) =====")
    for col in ["매수_피보4턴", "매수_피보5턴", "매수_엔하4턴", "매수_엔하5턴"]:
        print(f"  {col}: {int(df[col].sum())}건")
    print(f"  매수신호(통합, OR): {int(df['매수신호'].sum())}건")
    print(f"  매도신호: {int(df['매도신호'].sum())}건")


def print_round_summary(trades):
    print("\n===== 라운드별 성과 =====")
    m = evaluate_trades(trades)
    cost_trades = trades.copy()
    cost_trades["손익률(%)"] = cost_trades["손익률(%)"] - ROUND_TRIP_COST_PCT
    m_cost = evaluate_trades(cost_trades)

    avg_hold = trades["보유봉수"].mean()
    print(f"  라운드 수: {m['거래수']}건")
    print(f"  승률: {m['승률(%)']:.1f}%")
    print(f"  손익비: {m['손익비']:.2f} (비용반영: {m_cost['손익비']:.2f})")
    print(f"  평균 보유: {avg_hold:.1f}봉(=시간) (약 {avg_hold / 7:.1f}거래일, 1거래일≈7봉 가정)")

    print("\n  매수사유별 건수:")
    print(trades["매수사유"].value_counts().to_string())

    print("\n  청산사유별 건수:")
    print(trades["청산사유"].value_counts().to_string())


def print_return_comparison(window_df, trades):
    print("\n===== 누적수익률: 롤링매매(복리, 비용반영) vs 단순보유 =====")

    equity = 1.0
    for pnl in trades["손익률(%)"]:
        equity *= 1 + (pnl - ROUND_TRIP_COST_PCT) / 100
    rolling_return_pct = (equity - 1) * 100

    buy_hold_return_pct = (window_df["Close"].iloc[-1] / window_df["Close"].iloc[0] - 1) * 100

    print(f"  구간: {window_df.index[0]} ~ {window_df.index[-1]}")
    print(f"  단순보유 수익률: {buy_hold_return_pct:+.2f}%")
    print(f"  롤링매매 누적수익률(복리, 비용반영): {rolling_return_pct:+.2f}%")
    print(f"  차이: {rolling_return_pct - buy_hold_return_pct:+.2f}%p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_WINDOW_START, help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_WINDOW_END, help="백테스트 종료일 (YYYY-MM-DD)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="결과 CSV 저장 경로")
    args = parser.parse_args()
    window_start, window_end, output_path = args.start, args.end, Path(args.output)

    print_phase_check(window_start, window_end)

    print(f"\n[INFO] {LABEL} 60분봉 로딩 중...")
    code, raw_min60 = fetch_kr_min60_bars(LABEL)
    if raw_min60.empty:
        print("[경고] 60분봉 데이터를 찾지 못했습니다.")
        return
    print(f"  전체 {len(raw_min60)}행, {raw_min60.index.min()} ~ {raw_min60.index.max()}")

    # 지표는 전체 기간으로 먼저 계산해서 워밍업(MA50/ATR20 등)이 백테스트
    # 구간 앞부분에서 끊기지 않게 한다 (기존 run_*.py 스크립트들과 동일한 원칙).
    full_df = add_all_indicators(raw_min60)
    window_df = full_df.loc[window_start:window_end].copy()
    if window_df.empty:
        print(f"[경고] {window_start}~{window_end} 구간에 데이터가 없습니다.")
        return
    print(f"[INFO] 백테스트 구간: {window_df.index[0]} ~ {window_df.index[-1]} ({len(window_df)}봉)")

    window_df = add_buy_sell_signals(window_df)
    print_signal_frequency(window_df)

    trades = simulate_rolling_trades(window_df)
    if trades.empty:
        print("[경고] 이 구간에서 라운드가 하나도 발생하지 않았습니다.")
        return

    trades.insert(0, "티커", code)
    trades.insert(0, "종목명", LABEL)

    DATA_DIR.mkdir(exist_ok=True)
    trades.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 총 {len(trades)}개 라운드를 저장했습니다: {output_path}")

    print_round_summary(trades)
    print_return_comparison(window_df, trades)


if __name__ == "__main__":
    main()
