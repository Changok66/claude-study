# -*- coding: utf-8 -*-
"""
엔상5/엔하5 반전각도 분석(run_reversal_angle_analysis.py)의 후속 검증 스크립트.

data/reversal_angle_analysis.csv에서 이미 확인한 "반전성공률"이 진짜로 의미가
있는 수치인지, 그리고 실제로 그 신호로 매매했다면 돈을 벌었을지를 추가로
확인한다. 확인 항목 4가지:

  1) 반전성공률(예: 66.8%)이 통계적으로 유의미한가?
     -> 귀무가설 "반전확률=50%(순전히 우연)"에 대한 이항검정(binomial test).
        src/ma50_wave.py의 _sign_test_pvalue()가 이미 이 계산(Binomial(n,0.5)
        양측검정)을 하고 있어서 그대로 재사용한다 (카이제곱도 결국 이 경우엔
        같은 결론을 주므로 따로 만들지 않는다).
  2) 실제로 매매했다면 승률뿐 아니라 손익비는 어땠을까?
     -> 엔상5 터치=숏 진입/엔하5 터치=롱 진입, "MA50 복귀 또는 N봉경과 중 먼저
        오는 조건"으로 청산하는 매매를 시뮬레이션한다
        (src/reversal_angle.py의 simulate_reversal_trades). 승률/손익비는
        src/ma50_backtest.py의 evaluate_trades()를 그대로 재사용한다.
  3) 종목별로도 방향이 일관되는가? (에이피알처럼 표본 10건 미만인 종목은 제외)
  4) 에이피알은 표본이 몇 건뿐인지 명시하고 판단을 보류한다.

결과 파일: data/reversal_angle_backtest.csv (거래 1건 = 1행, 종목/구간/밴드종류
포함). 통계검정/종목별 일관성/에이피알 안내는 콘솔에 요약으로 출력한다
(이 값들은 표 하나로 모으기엔 성격이 서로 달라서, 기존 스크립트들처럼 콘솔
요약으로 보여주는 방식을 그대로 따른다).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.ma50_backtest import evaluate_trades, split_train_test
from src.ma50_wave import _sign_test_pvalue
from src.reversal_angle import add_all_indicators, find_band_touch_events, simulate_reversal_trades

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "reversal_angle_backtest.csv"

N_BARS = 5  # 반전각도 분석과 동일한 값 (청산 조건의 "N봉"도 이 값을 그대로 쓴다)

# 종목별 결론을 낼 때 필요한 최소 표본 수. 기존 run_ma50_wave_domestic.py의
# MIN_TRADES_FOR_CONCLUSION=10과 같은 기준을 그대로 가져다 쓴다.
MIN_SAMPLE_FOR_CONCLUSION = 10

warnings = []


def build_stock_trades(label, ticker, df):
    """
    종목 1개를 받아서 지표 계산 -> 터치 이벤트 -> 학습/검증 구분 -> 매매 시뮬레이션
    까지 한 번에 처리하고, 거래 표(구간/종목 정보 포함)를 반환한다.
    이벤트가 없으면 None을 반환한다.
    """
    df = add_all_indicators(df)

    events = find_band_touch_events(df, n_bars=N_BARS)
    if events.empty:
        warnings.append(f"[{label}] 터치 이벤트가 없어 백테스트에서 제외합니다.")
        return None

    # 학습/검증 구분: run_reversal_angle_analysis.py, run_ma50_wave_domestic.py와
    # 동일한 방식 (원본 일봉을 시간순 70%/30%로 나눈 경계일 기준).
    train_part, _ = split_train_test(df, train_ratio=0.7)
    split_time = train_part.index[-1] if len(train_part) > 0 else df.index[-1]

    events = events.copy()
    events["구간"] = np.where(events["터치시각"] <= split_time, "학습", "검증")

    trades = simulate_reversal_trades(df, events, n_bars=N_BARS)
    # trades는 simulate_reversal_trades 안에서 events를 순서 그대로(행 순서 유지)
    # 훑으면서 만들어지므로, 같은 순서인 events["구간"] 값을 그대로 붙여도 된다.
    trades["구간"] = events["구간"].values
    trades.insert(0, "티커", ticker)
    trades.insert(0, "종목명", label)

    return trades


def print_significance_test(combined):
    """1) 반전성공률이 우연(50%)이 아니라고 볼 수 있는지 이항검정으로 확인."""
    print("\n===== 1) 반전성공률 통계적 유의성 (귀무가설: 반전확률=50%) =====")
    for (구간, 밴드종류), g in combined.groupby(["구간", "밴드종류"]):
        n = len(g)
        k = int(g["반전성공여부"].sum())
        p_value = _sign_test_pvalue(n, k)
        유의미 = p_value < 0.05
        print(
            f"[{구간}] {밴드종류}: 표본 {n}건, 성공 {k}건({k / n * 100:.1f}%), "
            f"p값={p_value:.4f} -> {'유의미(p<0.05)' if 유의미 else '유의미하지 않음'}"
        )


def print_backtest_performance(combined):
    """2) 실제 매매했다면 승률/손익비가 어땠을지 구간x밴드종류별 + 전체 합산으로 출력."""
    print("\n===== 2) 신호 매매 백테스트 성과 (승률/손익비) =====")
    for (구간, 밴드종류), g in combined.groupby(["구간", "밴드종류"]):
        m = evaluate_trades(g)
        print(
            f"[{구간}] {밴드종류}: 거래수={m['거래수']}, "
            f"승률={m['승률(%)']:.1f}%, 손익비={m['손익비']:.2f}"
        )

    overall = evaluate_trades(combined)
    print(
        f"[전체 합산] 9종목: 거래수={overall['거래수']}, "
        f"승률={overall['승률(%)']:.1f}%, 손익비={overall['손익비']:.2f}"
    )


def print_stock_consistency(combined):
    """3) 종목별로도 방향(반전각도 부호)이 일관되는지 확인 (에이피알 제외)."""
    print(f"\n===== 3) 종목별 일관성 (표본 {MIN_SAMPLE_FOR_CONCLUSION}건 미만은 제외) =====")

    # 밴드종류별로 "원래 기대하는 반전각도 부호"가 다르다.
    #   엔상5 터치(과열 -> 꺾임) : 반전각도가 음수여야 "예상대로"
    #   엔하5 터치(침체 -> 반등) : 반전각도가 양수여야 "예상대로"
    expected_sign = {"엔상5터치": -1, "엔하5터치": 1}

    for 밴드종류, sign in expected_sign.items():
        print(f"-- {밴드종류} (기대 반전각도 부호: {'음수' if sign < 0 else '양수'}) --")
        matched, checked = 0, 0

        for label in KR_STOCK_CODES:
            if label == "에이피알":
                continue  # 4번 항목에서 따로 다룬다.

            subset = combined[(combined["종목명"] == label) & (combined["밴드종류"] == 밴드종류)]
            n = len(subset)
            if n < MIN_SAMPLE_FOR_CONCLUSION:
                print(f"  {label}: {n}건 (표본부족, 판단 보류)")
                continue

            avg_angle = subset["반전각도(도)"].mean()
            success_rate = subset["반전성공여부"].mean() * 100
            direction_ok = (avg_angle * sign) > 0

            checked += 1
            if direction_ok:
                matched += 1

            print(
                f"  {label}: {n}건, 평균반전각도={avg_angle:+.1f}도, "
                f"반전성공률={success_rate:.1f}% -> {'예상방향 일치' if direction_ok else '예상방향과 반대'}"
            )

        if checked > 0:
            print(f"  => 판단 가능한 {checked}개 종목 중 {matched}개에서 예상 방향과 일치")
        else:
            print("  => 판단 가능한(표본 충분한) 종목이 없습니다.")


def print_apr_caveat(combined):
    """4) 에이피알은 표본이 몇 건뿐인지 명시하고 판단을 보류한다는 것을 확실히 알린다."""
    print("\n===== 4) 표본 부족 종목 안내 =====")
    apr = combined[combined["종목명"] == "에이피알"]
    if apr.empty:
        print("에이피알: 거래 자체가 없습니다.")
    else:
        by_band = apr.groupby("밴드종류").size().to_dict()
        print(
            f"에이피알: 총 {len(apr)}건({by_band}) - "
            f"기준치({MIN_SAMPLE_FOR_CONCLUSION}건) 미달로 표본부족입니다. "
            "이 종목에 대한 반전각도/승률/손익비 결론은 판단 보류합니다."
        )


def main():
    all_trades = []

    for label in KR_STOCK_CODES:
        ticker, df = fetch_kr_daily_bars(label)
        if df.empty:
            warnings.append(f"[{label}] 데이터를 받아오지 못했습니다.")
            continue

        print(f"[INFO] {label}({ticker}) 백테스트 처리 중... ({len(df)}행)")
        trades = build_stock_trades(label, ticker, df)
        if trades is not None:
            all_trades.append(trades)

    if not all_trades:
        print("[경고] 9종목 전부에서 거래가 발생하지 않았습니다. 결과 파일을 만들지 않습니다.")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values(["종목명", "터치시각"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 총 {len(combined)}건의 거래를 저장했습니다: {OUTPUT_PATH}")

    print_significance_test(combined)
    print_backtest_performance(combined)
    print_stock_consistency(combined)
    print_apr_caveat(combined)

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
