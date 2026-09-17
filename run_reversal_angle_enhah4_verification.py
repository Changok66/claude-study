# -*- coding: utf-8 -*-
"""
엔하4 롱 신호에, 엔하5 롱 신호 때 했던 것과 동일한 두 가지 재검증을 진행한다
(run_reversal_angle_enhah5_by_period.py와 완전히 같은 방식 - 밴드만 4번으로
바꿔서 재사용).

확인 항목 2가지:
  1) 실전 수수료(매수 0.015%/매도 0.165%) + 슬리피지(가정치, 편도 0.1%x2) =
     왕복 0.38%p를 반영해도 손익비가 유지되는지.
  2) 연도별/최근구간별 안정성 - 엔하5처럼 최근 표본이 사라지는 문제가 있는지.

data/reversal_angle_enha_3_4_5_comparison.csv에 이미 엔하4 거래 374건이
계산돼 있으므로, 다시 계산하지 않고 "밴드"=="엔하4"인 행만 걸러서 재사용한다.

결과 파일: data/reversal_angle_enhah4_verification.csv
"""

from pathlib import Path

import pandas as pd

from src.ma50_backtest import evaluate_trades
from src.ma50_wave import _sign_test_pvalue

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COMPARISON_PATH = DATA_DIR / "reversal_angle_enhah_3_4_5_comparison.csv"
OUTPUT_PATH = DATA_DIR / "reversal_angle_enhah4_verification.csv"

MIN_SAMPLE_FOR_REFERENCE = 10  # 연도별 표본이 이보다 적으면 "참고용"으로만 표시

# 엔하5 검증 때와 동일한 비용 가정 (run_reversal_angle_enhah5_by_period.py 참고)
BUY_FEE_PCT = 0.015
SELL_FEE_PCT = 0.165
SLIPPAGE_PCT_PER_SIDE = 0.1
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + 2 * SLIPPAGE_PCT_PER_SIDE


def load_enha4_baseline():
    """엔하3/4/5 비교 결과에서 엔하4 거래(374건)만 골라온다."""
    df = pd.read_csv(COMPARISON_PATH)
    enha4 = df[df["밴드"] == "엔하4"].copy()
    enha4["터치시각"] = pd.to_datetime(enha4["터치시각"])
    enha4["연도"] = enha4["터치시각"].dt.year
    return enha4


def add_cost_adjusted_pnl(df):
    """엔하4도 전부 롱 방향이므로 왕복비용을 그대로 손익률에서 빼면 된다."""
    df = df.copy()
    df["손익률_비용반영(%)"] = df["손익률(%)"] - ROUND_TRIP_COST_PCT
    return df


def print_yearly_table(df):
    print("\n===== 연도별 성과 (엔하4, 원본 손익률 기준) =====")
    for year, g in df.groupby("연도"):
        n = len(g)
        m = evaluate_trades(g)
        k = int(g["반전성공여부"].sum())
        p_value = _sign_test_pvalue(n, k)

        표본안내 = f"{n}건"
        if n < MIN_SAMPLE_FOR_REFERENCE:
            표본안내 += " [참고용]"

        print(
            f"  {year}년: 표본 {표본안내}, 승률={m['승률(%)']:.1f}%, "
            f"손익비={m['손익비']:.2f}, 반전성공률={k / n * 100:.1f}%(p={p_value:.4f})"
        )


def print_recent_gap_check(df):
    """엔하5처럼 최근 구간에 표본이 아예 사라지는 문제가 있는지 확인한다."""
    last_touch = df["터치시각"].max()
    print(f"\n===== 최근 구간 확인 (엔하5와 비교) =====")
    print(f"가장 최근 엔하4 터치: {last_touch.date()} (원본 가격 데이터는 2026-09-11까지 있음)")

    for years_back, label in [(1, "최근 1년"), (2, "최근 2년"), (3, "최근 3년")]:
        cutoff = pd.Timestamp.now(tz=df["터치시각"].dt.tz) - pd.DateOffset(years=years_back)
        recent = df[df["터치시각"] >= cutoff]
        n = len(recent)
        if n == 0:
            print(f"  {label}(오늘 기준 {cutoff.date()} 이후): {n}건")
            continue
        m = evaluate_trades(recent)
        print(f"  {label}(오늘 기준 {cutoff.date()} 이후): {n}건, 승률={m['승률(%)']:.1f}%, 손익비={m['손익비']:.2f}")

    recent1 = df[df["터치시각"] >= pd.Timestamp.now(tz=df["터치시각"].dt.tz) - pd.DateOffset(years=1)]
    if len(recent1) == 0:
        print("  -> 엔하5와 동일하게 최근 1년 표본이 없습니다.")
    else:
        print(
            f"  -> 엔하5와 달리 최근 1년에도 {len(recent1)}건의 터치가 있어, "
            "완전한 표본 공백은 아닙니다."
        )

    print("\n  최근 3개년(2023~2025) vs 그 이전 비교:")
    recent3 = df[df["연도"] >= 2023]
    older = df[df["연도"] < 2023]
    for label, g in [("2023년~현재", recent3), ("~2022년", older)]:
        n = len(g)
        if n == 0:
            print(f"    {label}: 표본 없음")
            continue
        m = evaluate_trades(g)
        표본안내 = f"{n}건" + (" [참고용]" if n < MIN_SAMPLE_FOR_REFERENCE else "")
        print(f"    {label}: 표본 {표본안내}, 승률={m['승률(%)']:.1f}%, 손익비={m['손익비']:.2f}")


def print_cost_adjusted_comparison(df):
    print(f"\n===== 실전 수수료/슬리피지 반영 결과 (엔하4) =====")
    print(
        f"가정: 매수수수료 {BUY_FEE_PCT}% + 매도수수료(세금포함) {SELL_FEE_PCT}% + "
        f"슬리피지(가정치, 편도 {SLIPPAGE_PCT_PER_SIDE}%x2) "
        f"= 왕복 총비용 {ROUND_TRIP_COST_PCT:.3f}%p (엔하5 검증 때와 동일한 가정)"
    )

    raw_metrics = evaluate_trades(df)
    cost_df = df.drop(columns=["손익률(%)"]).rename(columns={"손익률_비용반영(%)": "손익률(%)"})
    cost_metrics = evaluate_trades(cost_df)

    print(
        f"  [비용 반영 전] 거래수={raw_metrics['거래수']}, 승률={raw_metrics['승률(%)']:.1f}%, "
        f"손익비={raw_metrics['손익비']:.2f}"
    )
    print(
        f"  [비용 반영 후] 거래수={cost_metrics['거래수']}, 승률={cost_metrics['승률(%)']:.1f}%, "
        f"손익비={cost_metrics['손익비']:.2f}"
    )

    for 구간 in ["학습", "검증"]:
        g = df[df["구간"] == 구간]
        g_cost = cost_df[cost_df["구간"] == 구간]
        if len(g) == 0:
            continue
        m_raw = evaluate_trades(g)
        m_cost = evaluate_trades(g_cost)
        print(
            f"  [{구간}] 비용반영 전 손익비={m_raw['손익비']:.2f} -> "
            f"비용반영 후 손익비={m_cost['손익비']:.2f} (승률 {m_raw['승률(%)']:.1f}%->{m_cost['승률(%)']:.1f}%)"
        )


def main():
    enha4 = load_enha4_baseline()
    enha4 = add_cost_adjusted_pnl(enha4)

    DATA_DIR.mkdir(exist_ok=True)
    enha4.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 엔하4 {len(enha4)}건(+연도/비용반영손익률 컬럼)을 저장했습니다: {OUTPUT_PATH}")

    print_yearly_table(enha4)
    print_recent_gap_check(enha4)
    print_cost_adjusted_comparison(enha4)


if __name__ == "__main__":
    main()
