# -*- coding: utf-8 -*-
"""
"엔상3/4/5를 숏 진입이 아니라 롱 포지션 청산 타점으로 쓰면 어떨까"를
검증하는 스크립트.

진입은 그대로: 엔하4 또는 엔하5 터치 시점에 매수 (기존 find_band_touch_events
로 만든 이벤트를 그대로 재사용 - 이렇게 해야 "같은 진입, 다른 청산"을
공정하게 비교할 수 있다).

청산은 4가지 방식을 비교한다:
  1~3. 목표청산: 진입 이후 처음으로 엔상3/엔상4/엔상5를 터치하는 날 청산
       (src/reversal_angle.py의 simulate_target_exit_trades, 보유기간 상한 없음)
  4. 기존방식(비교용): MA50 복귀 또는 5봉 경과 중 먼저 오는 조건으로 청산
     (기존 simulate_reversal_trades를 그대로 재사용해서 다시 계산 - 이전
     세션 결과와 완전히 같은 숫자가 나와야 정상이다)

엔하4/엔하5 x 청산방식 4가지 = 8개 시나리오를 학습/검증 구간별로 비교한다.

결과 파일: data/enha_entry_ensah_exit_analysis.csv
(8개 시나리오의 거래를 모두 합친 표, "진입밴드"/"청산방식" 컬럼으로 구분)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_backtest import evaluate_trades, split_train_test
from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.reversal_angle import (
    add_all_indicators,
    find_band_touch_events,
    simulate_reversal_trades,
    simulate_target_exit_trades,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "enha_entry_ensah_exit_analysis.csv"

N_BARS = 5  # 진입 이벤트 탐지 + 기존방식 청산에 쓰는 값 (이전 세션과 동일)
ENTRY_BANDS = (4, 5)
TARGET_EXIT_BANDS = (3, 4, 5)

warnings = []


def build_entry_events(df, entry_band):
    """엔하{entry_band}터치 이벤트만 골라서 학습/검증 구분까지 붙여 반환한다."""
    band_label = f"엔하{entry_band}터치"

    events = find_band_touch_events(df, n_bars=N_BARS, band_no=entry_band)
    if events.empty:
        return pd.DataFrame()

    events = events[events["밴드종류"] == band_label].copy()
    if events.empty:
        return pd.DataFrame()

    train_part, _ = split_train_test(df, train_ratio=0.7)
    split_time = train_part.index[-1] if len(train_part) > 0 else df.index[-1]
    events["구간"] = np.where(events["터치시각"] <= split_time, "학습", "검증")

    return events


def process_stock(label, entry_band):
    """
    종목 1개, 진입밴드 1개에 대해 4가지 청산 시나리오(엔상3/4/5청산 +
    기존방식)의 거래 표를 만들어 리스트로 반환한다.
    """
    ticker, raw = fetch_kr_daily_bars(label)
    if raw.empty:
        warnings.append(f"[엔하{entry_band}][{label}] 데이터를 받아오지 못했습니다.")
        return []

    df = add_all_indicators(raw)
    events = build_entry_events(df, entry_band)
    if events.empty:
        return []

    frames = []

    # 1~3. 목표청산 (엔상3/4/5)
    for exit_band in TARGET_EXIT_BANDS:
        trades = simulate_target_exit_trades(df, events, exit_band)
        trades["구간"] = events["구간"].values
        trades.insert(0, "티커", ticker)
        trades.insert(0, "종목명", label)
        trades.insert(0, "진입밴드", f"엔하{entry_band}")
        frames.append(trades)

    # 4. 기존방식 (비교용, MA50복귀 또는 5봉경과)
    baseline = simulate_reversal_trades(df, events, n_bars=N_BARS)
    baseline["구간"] = events["구간"].values
    baseline["청산방식"] = "기존방식(MA50복귀or5봉)"
    baseline.insert(0, "티커", ticker)
    baseline.insert(0, "종목명", label)
    baseline.insert(0, "진입밴드", f"엔하{entry_band}")
    # simulate_reversal_trades 결과에는 "밴드종류"/"반전각도(도)"/"반전성공여부"
    # 컬럼이 더 있는데, 이번 비교표에는 필요 없는 컬럼이라 통일을 위해 뺀다.
    baseline = baseline.drop(columns=["밴드종류", "반전각도(도)", "반전성공여부"])
    frames.append(baseline)

    return frames


def print_comparison_table(combined):
    print("\n===== 진입밴드 x 청산방식 비교표 =====")
    header = (
        f"{'진입밴드':<8}{'청산방식':<22}{'표본수':>8}{'평균보유봉수':>12}"
        f"{'승률(학습)':>12}{'손익비(학습)':>14}{'승률(검증)':>12}{'손익비(검증)':>14}"
    )
    print(header)

    for (진입밴드, 청산방식), g in combined.groupby(["진입밴드", "청산방식"], sort=False):
        n = len(g)
        avg_hold = g["보유봉수"].mean()

        row_str = f"{진입밴드:<8}{청산방식:<22}{n:>8}{avg_hold:>12.1f}"
        for 구간 in ["학습", "검증"]:
            sub = g[g["구간"] == 구간]
            if len(sub) == 0:
                row_str += f"{'표본없음':>12}{'':>14}"
                continue
            m = evaluate_trades(sub)
            row_str += f"{m['승률(%)']:>11.1f}%{m['손익비']:>14.2f}"
        print(row_str)


def print_exit_reason_breakdown(combined):
    print("\n===== 목표청산 시나리오의 '도달률' 확인 =====")
    target_rows = combined[combined["청산방식"].str.contains("청산$", regex=True) & ~combined["청산방식"].str.startswith("기존")]
    for (진입밴드, 청산방식), g in target_rows.groupby(["진입밴드", "청산방식"], sort=False):
        n = len(g)
        도달 = (g["청산사유"] != "미도달(데이터종료)").sum()
        print(f"  {진입밴드} -> {청산방식}: {n}건 중 {도달}건 목표 도달 ({도달 / n * 100:.1f}%), 평균보유봉수={g['보유봉수'].mean():.1f}")


def main():
    all_frames = []

    for entry_band in ENTRY_BANDS:
        print(f"\n[INFO] 엔하{entry_band} 진입 기준 계산 중 (9종목 x 4개 청산방식)...")
        for label in KR_STOCK_CODES:
            frames = process_stock(label, entry_band)
            all_frames.extend(frames)

    if not all_frames:
        print("[경고] 거래가 하나도 발생하지 않았습니다.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["진입밴드", "청산방식", "종목명", "터치시각"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 총 {len(combined)}건을 저장했습니다: {OUTPUT_PATH}")

    print_comparison_table(combined)
    print_exit_reason_breakdown(combined)

    print(
        "\n[참고] 진입밴드 2개 x 청산방식 4개 = 8개 조합을 비교하는 것 자체가 "
        "다중비교입니다. 숫자만 보고 가장 좋아 보이는 조합 하나를 바로 채택하지 "
        "말고, 학습/검증 양쪽에서 같은 방향(예: 목표가가 멀수록 손익비가 어떻게 "
        "변하는지)이 일관되게 나오는지부터 확인하는 것을 권장합니다."
    )

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
