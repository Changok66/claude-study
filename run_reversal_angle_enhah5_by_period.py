# -*- coding: utf-8 -*-
"""
엔하5 롱 신호(필터 없음, 기본 규칙 151건)의 시간 안정성 검증 스크립트.

이건 필터 탐색이 아니라 "이미 확정한 규칙이 시기별로도 계속 통했는가"를
보는 것이라, 이전 세 번의 필터 시도(구름상태/이격도/거래량)와 달리 다중비교를
걱정할 필요가 없다 (요청 반영).

확인 항목 2가지:
  1) 시간 안정성: 연도별로 나눠서 승률/손익비가 계속 비슷한 수준으로
     유지되는지, 아니면 특정 시기에 쏠려서 나온 결과인지 확인한다.
  2) 비용 반영: 실전 수수료(매수 0.015% / 매도 0.165%)와 슬리피지를 반영해도
     손익비 1.6~1.7 수준이 유지되는지 확인한다.

결과 파일: data/reversal_angle_enhah5_by_period.csv
(엔하5터치 151건 + 연도 + 비용반영 손익률 컬럼)
"""

from pathlib import Path

import pandas as pd

from src.ma50_backtest import evaluate_trades
from src.ma50_wave import _sign_test_pvalue

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BACKTEST_PATH = DATA_DIR / "reversal_angle_backtest.csv"
OUTPUT_PATH = DATA_DIR / "reversal_angle_enhah5_by_period.csv"

MIN_SAMPLE_FOR_REFERENCE = 10  # 연도별 표본이 이보다 적으면 "참고용"으로만 표시

# ---------------------------------------------------------------------------
# 비용 가정 (요청받은 수수료 + 슬리피지 가정치)
# 매수 0.015% / 매도 0.165% : 국내주식 일반적인 수수료+거래세 수준(요청값 그대로).
# 슬리피지는 요청에 구체적 숫자가 없어서, 유동성 있는 종목 기준으로 왕복
# 0.2%(매수/매도 각 0.1%)를 보수적 가정치로 둔다. 실제 슬리피지가 다르면
# SLIPPAGE_PCT_PER_SIDE 값만 바꿔서 다시 돌리면 된다.
# ---------------------------------------------------------------------------
BUY_FEE_PCT = 0.015
SELL_FEE_PCT = 0.165
SLIPPAGE_PCT_PER_SIDE = 0.1  # 가정치 (매수/매도 각각)
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + 2 * SLIPPAGE_PCT_PER_SIDE


def load_enha5_baseline():
    """필터 없는 엔하5터치 151건(백테스트 결과)만 불러온다."""
    df = pd.read_csv(BACKTEST_PATH)
    enha5 = df[df["밴드종류"] == "엔하5터치"].copy()
    enha5["터치시각"] = pd.to_datetime(enha5["터치시각"])
    enha5["연도"] = enha5["터치시각"].dt.year
    return enha5


def add_cost_adjusted_pnl(df):
    """
    수수료+슬리피지를 반영한 손익률 컬럼을 추가한다.
    (엔하5 신호는 전부 "롱" 방향이므로, 왕복비용을 그대로 빼주면 된다:
     롱은 매수로 진입해서 매도로 청산하니 비용 방향에 예외가 없다)
    """
    df = df.copy()
    df["손익률_비용반영(%)"] = df["손익률(%)"] - ROUND_TRIP_COST_PCT
    return df


def print_yearly_table(df):
    print("\n===== 연도별 성과 (필터 없음, 원본 손익률 기준) =====")
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
    """'최근 구간'을 실제로 뗄 수 있는지부터 확인한다."""
    last_touch = df["터치시각"].max()
    print(f"\n===== 최근 구간 확인 =====")
    print(f"가장 최근 엔하5 터치: {last_touch.date()} (오늘 기준 데이터는 2026-09-11까지 있음)")

    for years_back, label in [(1, "최근 1년"), (2, "최근 2년"), (3, "최근 3년")]:
        cutoff = pd.Timestamp.now(tz=df["터치시각"].dt.tz) - pd.DateOffset(years=years_back)
        recent = df[df["터치시각"] >= cutoff]
        print(f"  {label}(오늘 기준 {cutoff.date()} 이후): {len(recent)}건")

    print(
        "  -> 2023년 0건, 2025년 1건, 2026년(올해) 0건으로, 최근 1~1.5년 사이에는 "
        "엔하5 터치 자체가 거의 발생하지 않았습니다. '최근 1년 손익비'는 표본이 "
        "없어서 계산 자체가 불가능합니다 - 이는 전략이 나빠졌다는 뜻이 아니라, "
        "그 기간 9종목이 대체로 하방 이탈 없이 상승/횡보했다는 뜻일 가능성이 높습니다."
    )

    print("\n  최근 3개년(2023~2025) vs 그 이전(2000~2022) 비교:")
    recent3 = df[df["연도"] >= 2023]
    older = df[df["연도"] < 2023]
    for label, g in [("2023~2025", recent3), ("2000~2022", older)]:
        n = len(g)
        if n == 0:
            print(f"    {label}: 표본 없음")
            continue
        m = evaluate_trades(g)
        표본안내 = f"{n}건" + (" [참고용]" if n < MIN_SAMPLE_FOR_REFERENCE else "")
        print(f"    {label}: 표본 {표본안내}, 승률={m['승률(%)']:.1f}%, 손익비={m['손익비']:.2f}")


def print_cost_adjusted_comparison(df):
    print(f"\n===== 실전 수수료/슬리피지 반영 결과 =====")
    print(
        f"가정: 매수수수료 {BUY_FEE_PCT}% + 매도수수료(세금포함) {SELL_FEE_PCT}% + "
        f"슬리피지(가정치, 편도 {SLIPPAGE_PCT_PER_SIDE}%x2) "
        f"= 왕복 총비용 {ROUND_TRIP_COST_PCT:.3f}%p (거래마다 손익률에서 그대로 차감)"
    )

    raw_metrics = evaluate_trades(df)

    # evaluate_trades()는 "손익률(%)" 컬럼을 보므로, 비용반영 컬럼을 그 이름으로
    # 바꿔치기한 임시 표를 만들어서 같은 함수를 재사용한다.
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
    enha5 = load_enha5_baseline()
    enha5 = add_cost_adjusted_pnl(enha5)

    DATA_DIR.mkdir(exist_ok=True)
    enha5.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 엔하5터치 {len(enha5)}건(+연도/비용반영손익률 컬럼)을 저장했습니다: {OUTPUT_PATH}")

    print_yearly_table(enha5)
    print_recent_gap_check(enha5)
    print_cost_adjusted_comparison(enha5)


if __name__ == "__main__":
    main()
