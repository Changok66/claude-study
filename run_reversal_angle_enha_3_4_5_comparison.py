# -*- coding: utf-8 -*-
"""
엔하3/엔하4 터치도 엔하5와 같은 방식으로 매수 타점 후보로 검증하고,
엔하3 vs 엔하4 vs 엔하5를 나란히 비교하는 스크립트.

재사용: src/reversal_angle.py의 find_band_touch_events()/simulate_reversal_trades()
가 이제 밴드 번호(band_no)를 파라미터로 받도록 일반화돼 있어서(기존에는 5번선에
고정돼 있었다), 로직을 새로 만들지 않고 band_no만 3/4로 바꿔서 그대로 재사용한다.

엔하5는 "이미 검증된 결과"이므로 다시 계산하지 않고
data/reversal_angle_backtest.csv에 저장된 151건을 그대로 참고용으로 가져다 쓴다
(요청 반영 - 재계산 시 미세한 부동소수점 차이 등으로 숫자가 달라 보이는 걸
방지하기 위함이기도 하다).

주의(요청 반영): 밴드 3개를 나란히 비교하는 것 자체가 다중비교다. 그래서
"가장 좋아 보이는 밴드 하나"를 덥석 채택하지 않고, 안쪽(3)->바깥쪽(5)으로
갈수록 표본수/승률/손익비가 어떤 경향을 보이는지 전체 패턴을 같이 살펴본다.

결과 파일: data/reversal_angle_enhah_3_4_5_comparison.csv
(엔하3/엔하4/엔하5 거래 전체를 하나로 합친 표, "밴드" 컬럼으로 구분)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_backtest import evaluate_trades, split_train_test
from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.ma50_wave import _sign_test_pvalue
from src.reversal_angle import add_all_indicators, find_band_touch_events, simulate_reversal_trades

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BACKTEST_PATH = DATA_DIR / "reversal_angle_backtest.csv"  # 엔하5 기존 결과 재사용
OUTPUT_PATH = DATA_DIR / "reversal_angle_enhah_3_4_5_comparison.csv"

N_BARS = 5
NEW_BANDS = (3, 4)  # 이번에 새로 계산할 밴드 (엔하5는 기존 결과 재사용)

warnings = []


def build_band_trades(band_no):
    """
    밴드 번호 하나(3 또는 4)를 받아서, 9종목 전체에 대해 엔하{band_no}터치
    거래 표를 만들어 반환한다 (run_reversal_angle_backtest.py의 로직을
    band_no만 바꿔서 그대로 재사용한 것).
    """
    band_label = f"엔하{band_no}터치"
    all_trades = []

    for label in KR_STOCK_CODES:
        ticker, raw = fetch_kr_daily_bars(label)
        if raw.empty:
            warnings.append(f"[엔하{band_no}][{label}] 데이터를 받아오지 못했습니다.")
            continue

        df = add_all_indicators(raw)
        events = find_band_touch_events(df, n_bars=N_BARS, band_no=band_no)
        if events.empty:
            continue

        # 이번 분석은 매수 타점(하단 밴드)만 보는 것이므로 엔하N터치만 남긴다.
        events = events[events["밴드종류"] == band_label].copy()
        if events.empty:
            continue

        # 학습/검증 구분: 기존 스크립트들과 동일한 방식.
        train_part, _ = split_train_test(df, train_ratio=0.7)
        split_time = train_part.index[-1] if len(train_part) > 0 else df.index[-1]
        events["구간"] = np.where(events["터치시각"] <= split_time, "학습", "검증")

        trades = simulate_reversal_trades(df, events, n_bars=N_BARS)
        trades["구간"] = events["구간"].values
        trades.insert(0, "티커", ticker)
        trades.insert(0, "종목명", label)
        trades.insert(0, "밴드", f"엔하{band_no}")

        all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()

    return pd.concat(all_trades, ignore_index=True)


def load_existing_enha5():
    """이미 검증해둔 엔하5 결과(data/reversal_angle_backtest.csv)를 그대로 가져온다."""
    df = pd.read_csv(BACKTEST_PATH)
    enha5 = df[df["밴드종류"] == "엔하5터치"].copy()
    enha5.insert(0, "밴드", "엔하5")
    return enha5


def evaluate_band(df, band_label):
    """밴드 1개에 대해 표본수/유의성/학습·검증 승률·손익비를 출력하고, 요약행을 반환한다."""
    print(f"\n--- {band_label} ---")
    n_total = len(df)
    k_total = int(df["반전성공여부"].sum())
    p_total = _sign_test_pvalue(n_total, k_total)
    print(
        f"  전체 표본 {n_total}건, 반전성공률 {k_total / n_total * 100:.1f}% "
        f"(p={p_total:.4f}, 귀무가설 50%)"
    )

    summary_row = {"밴드": band_label, "표본수": n_total}

    for 구간 in ["학습", "검증"]:
        g = df[df["구간"] == 구간]
        n = len(g)
        if n == 0:
            print(f"  [{구간}] 표본 없음")
            summary_row[f"승률_{구간}(%)"] = np.nan
            summary_row[f"손익비_{구간}"] = np.nan
            continue

        m = evaluate_trades(g)
        print(f"  [{구간}] 표본 {n}건, 승률={m['승률(%)']:.1f}%, 손익비={m['손익비']:.2f}")
        summary_row[f"승률_{구간}(%)"] = m["승률(%)"]
        summary_row[f"손익비_{구간}"] = m["손익비"]

    return summary_row


def print_comparison_table(summary_rows):
    print("\n===== 엔하3 vs 엔하4 vs 엔하5 비교표 =====")
    header = f"{'밴드':<8}{'표본수':>8}{'승률(학습)':>12}{'손익비(학습)':>14}{'승률(검증)':>12}{'손익비(검증)':>14}"
    print(header)
    for row in summary_rows:
        print(
            f"{row['밴드']:<8}{row['표본수']:>8}"
            f"{row.get('승률_학습(%)', float('nan')):>11.1f}%"
            f"{row.get('손익비_학습', float('nan')):>14.2f}"
            f"{row.get('승률_검증(%)', float('nan')):>11.1f}%"
            f"{row.get('손익비_검증', float('nan')):>14.2f}"
        )


def print_trend_discussion(summary_rows):
    """
    안쪽(엔하3) -> 바깥쪽(엔하5)로 갈수록 표본수/손익비가 어떻게 바뀌는지
    전체적인 경향을 설명한다 (다중비교 위험 때문에 "1등만 채택" 하지 않기 위함).
    """
    print("\n===== 전체 경향 (다중비교 주의 - 어느 한 밴드만 덥석 채택하지 않음) =====")

    by_band = {row["밴드"]: row for row in summary_rows}
    order = ["엔하3", "엔하4", "엔하5"]

    표본수_추이 = [by_band[b]["표본수"] for b in order if b in by_band]
    표본수_문구 = " -> ".join(f"{b}={by_band[b]['표본수']}건" for b in order if b in by_band)
    print(f"  표본수 추이(안쪽->바깥쪽): {표본수_문구}")
    if len(표본수_추이) >= 2 and 표본수_추이[0] > 표본수_추이[-1]:
        print("  -> 예상대로 안쪽(엔하3)일수록 터치가 훨씬 자주 발생, 바깥쪽(엔하5)일수록 희귀함.")

    손익비_학습_추이 = [by_band[b].get("손익비_학습", float("nan")) for b in order if b in by_band]
    손익비_검증_추이 = [by_band[b].get("손익비_검증", float("nan")) for b in order if b in by_band]
    print(
        "  손익비(학습) 추이(안쪽->바깥쪽): "
        + " -> ".join(f"{v:.2f}" for v in 손익비_학습_추이)
    )
    print(
        "  손익비(검증) 추이(안쪽->바깥쪽): "
        + " -> ".join(f"{v:.2f}" for v in 손익비_검증_추이)
    )

    print(
        "\n  판단 기준: 표본이 충분히 많으면서(수백 건 단위) 학습/검증 손익비가 "
        "둘 다 1을 안정적으로 넘고 서로 크게 어긋나지 않는 밴드를 '표본-성과 균형점'"
        "으로 본다. 여러 밴드 중 숫자만 보고 가장 높은 손익비를 고르는 것이 아니라, "
        "이 추이 자체(안쪽일수록 표본은 늘고 손익비는 어떻게 변하는지)를 근거로 "
        "판단해야 다중비교 함정을 피할 수 있다."
    )


def main():
    all_frames = []
    summary_rows = []

    for band_no in NEW_BANDS:
        print(f"\n[INFO] 엔하{band_no} 계산 중 (9종목)...")
        band_trades = build_band_trades(band_no)
        if band_trades.empty:
            warnings.append(f"엔하{band_no}: 터치 이벤트가 하나도 없습니다.")
            continue
        all_frames.append(band_trades)
        summary_rows.append(evaluate_band(band_trades, f"엔하{band_no}"))

    enha5 = load_existing_enha5()
    all_frames.append(enha5)
    summary_rows.append(evaluate_band(enha5, "엔하5 (기존 검증 결과, 참고용)"))
    # 비교표/추이 계산에서는 "엔하5"로 통일해서 쓴다.
    summary_rows[-1]["밴드"] = "엔하5"

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["밴드", "종목명", "터치시각"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 엔하3/엔하4/엔하5 합쳐서 총 {len(combined)}건을 저장했습니다: {OUTPUT_PATH}")

    print_comparison_table(summary_rows)
    print_trend_discussion(summary_rows)

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
