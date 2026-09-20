# -*- coding: utf-8 -*-
"""
피보4/피보5가 새 추세선(수식4=MA13+MA19 평균, 수식5=MA15+MA21 평균)을 돌파하는
크로스오버 전략을 검증하는 실행 스크립트다. "피보 4 또는 5", "추세선 수식4
또는 5" 둘 다 확정되지 않아서, 4가지 조합을 전부 계산해서 비교한다.

기본 실행(인자 없음): 삼성전자 하나만 처리해서 조합별 신호 건수/최근 거래
몇 건을 콘솔에 보여준다(먼저 확인받기 위함, 파일 저장 안 함).
`--all` 옵션: 9종목 전체 처리 -> 조합별 x 구간별(학습/검증) 승률/손익비/
비용반영손익비/평균보유봉수 비교표 + 기존 엔하2+피보턴과 비교 ->
data/fibo_trendline_breakout_backtest.csv 저장.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.fibo_trendline_breakout import FIBO_COLS, MA_PAIRS, add_trend_lines, find_breakout_trades
from src.ma50_backtest import evaluate_trades, split_train_test
from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.reversal_angle import add_fibo_5_8_13

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "fibo_trendline_breakout_backtest.csv"
EXISTING_COMPARISON_PATH = DATA_DIR / "enha2_fibo_turn_backtest.csv"

# 이전 세션(엔하2+피보턴 검증)과 동일한 실전비용 가정.
BUY_FEE_PCT = 0.015
SELL_FEE_PCT = 0.165
SLIPPAGE_PCT_PER_SIDE = 0.1
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + 2 * SLIPPAGE_PCT_PER_SIDE

MIN_SAMPLE_FOR_CONCLUSION = 30

# 가능한 4가지 조합: (피보4,추세선4) (피보4,추세선5) (피보5,추세선4) (피보5,추세선5)
COMBOS = [(fibo_col, f"추세선{name[-1]}") for fibo_col in FIBO_COLS for name in MA_PAIRS]

warnings = []


def prepare_stock_df(label):
    ticker, raw = fetch_kr_daily_bars(label)
    if raw.empty:
        return None, None
    df = add_fibo_5_8_13(raw)
    df = add_trend_lines(df)
    return ticker, df


def run_one_stock(label, ticker, df):
    """종목 1개에 대해 4가지 조합 전부 계산해서 하나의 표로 합친다."""
    all_trades = []
    for fibo_col, trend_col in COMBOS:
        trades = find_breakout_trades(df, fibo_col, trend_col)
        if trades.empty:
            continue
        trades.insert(0, "조합", f"{fibo_col}x{trend_col}")
        trades.insert(0, "티커", ticker)
        trades.insert(0, "종목명", label)
        all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True)


def print_samsung_preview(trades):
    print("\n===== 삼성전자 조합별 신호 건수 =====")
    for combo, g in trades.groupby("조합"):
        print(f"{combo}: {len(g)}건")

    print("\n===== 표본 확인용 - 조합별 최근 3건 =====")
    for combo, g in trades.groupby("조합"):
        print(f"\n--- {combo} ---")
        preview = g.sort_values("진입일").tail(3)[
            ["진입일", "진입가", "청산일", "청산가", "청산사유", "손익률(%)"]
        ].copy()
        # 날짜 컬럼이 섞여 있으면 DataFrame.round()가 경고를 내므로, 숫자
        # 컬럼만 따로 반올림한다.
        for col in ["진입가", "청산가", "손익률(%)"]:
            preview[col] = preview[col].round(2)
        print(preview.to_string(index=False))


def summarize(trades_df, label):
    """구간(학습/검증)별 표본수/승률/손익비/비용반영손익비/평균보유봉수/연환산근사를 계산."""
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

        rows.append({
            "신호": label, "구간": 구간, "표본수": n,
            "승률(%)": m["승률(%)"], "손익비": m["손익비"],
            "손익비_비용반영": m_cost["손익비"],
            "평균보유봉수": avg_hold, "연환산근사(%)": annualized,
        })
    return rows


def load_existing_enha2():
    if not EXISTING_COMPARISON_PATH.exists():
        warnings.append(f"{EXISTING_COMPARISON_PATH} 파일이 없어 엔하2+피보턴 비교를 건너뜁니다.")
        return None
    return pd.read_csv(EXISTING_COMPARISON_PATH, encoding="utf-8-sig")


def print_comparison_table(combined):
    print("\n===== 조합별 x 구간별 비교표 =====")
    summary_rows = []
    for combo, g in combined.groupby("조합"):
        summary_rows.extend(summarize(g, combo))

    existing = load_existing_enha2()
    if existing is not None:
        summary_rows.extend(summarize(existing, "엔하2+피보턴(기존)"))

    summary = pd.DataFrame(summary_rows)
    cols = ["신호", "구간", "표본수", "승률(%)", "손익비", "손익비_비용반영", "평균보유봉수", "연환산근사(%)"]
    print(summary[cols].round(2).to_string(index=False))
    print(f"\n(표본 {MIN_SAMPLE_FOR_CONCLUSION}건 미만은 참고용. 실전비용은 왕복 0.38%p 적용)")
    print(
        "\n[주의] 조합 4개 + 기존 엔하2+피보턴까지 여러 개를 나란히 비교했으므로 "
        "다중비교 위험이 있습니다. 표본이 충분하면서 학습/검증 손익비가 둘 다 "
        "비용반영 후에도 1을 안정적으로 넘는지를 기준으로 판단하세요."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="9종목 전체 실행 (기본은 삼성전자만 테스트)")
    args = parser.parse_args()

    if not args.all:
        label = "삼성전자"
        print(f"[INFO] {label} 단독 테스트 실행 중...")
        ticker, df = prepare_stock_df(label)
        if df is None:
            print("[경고] 데이터를 받아오지 못했습니다.")
            return
        trades = run_one_stock(label, ticker, df)
        if trades.empty:
            print("[경고] 4개 조합 전부에서 신호가 하나도 없습니다.")
            return
        print_samsung_preview(trades)
        print("\n[INFO] 확인 후 9종목 전체로 진행하려면 --all 옵션을 붙여서 다시 실행하세요.")
        return

    all_trades = []
    for label in KR_STOCK_CODES:
        print(f"[INFO] {label} 처리 중...")
        ticker, df = prepare_stock_df(label)
        if df is None:
            warnings.append(f"[{label}] 데이터를 받아오지 못했습니다.")
            continue

        trades = run_one_stock(label, ticker, df)
        if trades.empty:
            continue

        train_part, _ = split_train_test(df, train_ratio=0.7)
        split_time = train_part.index[-1] if len(train_part) > 0 else df.index[-1]
        trades["구간"] = np.where(trades["진입일"] <= split_time, "학습", "검증")
        all_trades.append(trades)

    if not all_trades:
        print("[경고] 9종목 전부에서 신호가 없습니다. 결과 파일을 만들지 않습니다.")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values(["조합", "종목명", "진입일"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 총 {len(combined)}건을 저장했습니다: {OUTPUT_PATH}")

    print_comparison_table(combined)

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
