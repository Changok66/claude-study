# -*- coding: utf-8 -*-
"""
엔하5 롱 신호(151건)에 "거래량 필터"를 걸어서 손익비가 개선되는지 확인하는
스크립트. (구름상태/피보일목 -> 이격도 크기에 이은 세 번째 필터 시도)

거래량 정의(요청 그대로): 터치 시점 봉의 거래량을, 그 직전 N=20봉 평균
거래량과 비교한 비율.
  거래량비율 = 터치일 거래량 / 직전 20봉(터치일 제외) 평균 거래량
"직전 20봉 제외"로 계산하는 이유: 터치일 자기 자신의 거래량을 자기 자신의
평균에 포함시키면(그날 거래량이 유난히 크면 평균도 같이 커져서) 비율이
과소평가되는 왜곡이 생긴다. 그래서 "어제까지의 평균"과 "오늘"을 비교한다.

data/reversal_angle_analysis.csv, data/reversal_angle_backtest.csv에는
거래량(Volume) 정보가 없어서, src/ma50_data.py로 종목별 원본 일봉을 다시
받아 거래량비율만 새로 계산한 뒤, 기존 엔하5 이벤트 151건에 붙인다.

방법 (이격도 크기 분석과 동일한 방식):
  1. 거래량비율을 3분위(적음/보통/많음)로 나눈다. 경계는 학습구간 데이터만
     으로 정해서 학습/검증에 동일하게 적용한다 (미래 정보 편향 방지).
  2. 구간(적음/보통/많음) x 학습/검증 별로 승률/손익비를 비교한다.
  3. 학습/검증에서 같은 방향(적음<보통<많음 등)이 재현되는지 확인한다.

중요한 주의사항 (요청 반영): 이번이 벌써 세 번째 필터 시도다(구름상태/피보일목
-> 실패, 이격도 크기 -> 실패). 필터를 계속 이것저것 시도하다 보면 우연히
그럴듯해 보이는 결과가 나올 위험(다중비교 문제)이 커진다. 그래서 이번 결과도
학습/검증 양쪽에서 명확하고 일관된 개선이 확인되지 않으면, 바로 채택하지
않고 "여기서 필터 탐색을 중단한다"는 결론을 내는 것을 기본값으로 삼는다.

결과 파일: data/reversal_angle_enhah5_by_volume.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_backtest import evaluate_trades
from src.ma50_data import fetch_kr_daily_bars
from src.ma50_wave import _sign_test_pvalue

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ANALYSIS_PATH = DATA_DIR / "reversal_angle_analysis.csv"
BACKTEST_PATH = DATA_DIR / "reversal_angle_backtest.csv"
OUTPUT_PATH = DATA_DIR / "reversal_angle_enhah5_by_volume.csv"

VOLUME_LOOKBACK = 20  # "최근 N봉" 평균 거래량 계산에 쓰는 N (요청 예시값)
MIN_SAMPLE_FOR_CONCLUSION = 20  # 이격도 분석 때와 동일한 기준
JOIN_KEYS = ["종목명", "티커", "터치시각", "밴드종류"]
BUCKET_LABELS = ["적음", "보통", "많음"]


def load_merged_enha5():
    """두 결과 CSV를 합쳐서 엔하5터치 행만 골라 반환한다."""
    analysis = pd.read_csv(ANALYSIS_PATH)
    backtest = pd.read_csv(BACKTEST_PATH)

    analysis_slim = analysis[JOIN_KEYS]  # 이번엔 참고 컬럼 없이 조인키만 있으면 충분
    merged = backtest.merge(analysis_slim, on=JOIN_KEYS, how="left")
    enha5 = merged[merged["밴드종류"] == "엔하5터치"].copy()
    return enha5


def add_volume_ratio(enha5_df, n_bars=VOLUME_LOOKBACK):
    """
    종목별로 원본 일봉을 다시 받아서 "거래량비율" 컬럼을 계산해 붙인다.
    (거래량비율 = 터치일 거래량 / 터치일 제외 직전 n_bars봉 평균 거래량)
    """
    enha5_df = enha5_df.copy()
    enha5_df["거래량비율"] = np.nan

    for label in enha5_df["종목명"].unique():
        _, raw = fetch_kr_daily_bars(label)

        # shift(1) : "오늘을 포함하지 않은, 어제까지의" 평균이 되도록 한 칸 밀어준다.
        avg_volume_prior = raw["Volume"].rolling(window=n_bars).mean().shift(1)
        volume_ratio = raw["Volume"] / avg_volume_prior

        stock_rows = enha5_df.index[enha5_df["종목명"] == label]
        for row_idx in stock_rows:
            touch_ts = pd.to_datetime(enha5_df.loc[row_idx, "터치시각"])
            pos = raw.index.get_loc(touch_ts)
            enha5_df.loc[row_idx, "거래량비율"] = volume_ratio.iloc[pos]

    return enha5_df


def add_volume_bucket(df):
    """
    학습구간 거래량비율만으로 3분위 경계를 구해서, 전체(학습+검증)에 동일하게
    적용한 "거래량구간"(적음/보통/많음) 컬럼을 추가한다.
    """
    df = df.copy()
    train_values = df.loc[df["구간"] == "학습", "거래량비율"].dropna()
    cut1, cut2 = np.percentile(train_values, [100 / 3, 200 / 3])

    def classify(v):
        if pd.isna(v):
            return None
        if v <= cut1:
            return "적음"
        elif v <= cut2:
            return "보통"
        else:
            return "많음"

    df["거래량구간"] = df["거래량비율"].apply(classify)
    return df, cut1, cut2


def print_bucket_performance(df, cut1, cut2):
    print(
        f"\n[INFO] 학습구간 기준 3분위 경계 -> 적음: 거래량비율<={cut1:.2f}배, "
        f"보통: {cut1:.2f}~{cut2:.2f}배, 많음: {cut2:.2f}배 초과"
    )

    pf_by_group = {}

    print("\n===== 거래량구간 x 학습/검증 별 성과 =====")
    for 구간 in ["학습", "검증"]:
        for bucket in BUCKET_LABELS:
            g = df[(df["구간"] == 구간) & (df["거래량구간"] == bucket)]
            n = len(g)

            if n == 0:
                print(f"  [{구간}] {bucket}: 표본 없음")
                continue

            m = evaluate_trades(g)
            k = int(g["반전성공여부"].sum())
            p_value = _sign_test_pvalue(n, k)

            표본안내 = f"{n}건"
            if n < MIN_SAMPLE_FOR_CONCLUSION:
                표본안내 += f" [참고용 - {MIN_SAMPLE_FOR_CONCLUSION}건 미만이라 확정 결론 보류]"

            print(
                f"  [{구간}] {bucket}: 표본 {표본안내}, 승률={m['승률(%)']:.1f}%, "
                f"손익비={m['손익비']:.2f}, 반전성공률={k / n * 100:.1f}%(p={p_value:.4f})"
            )
            pf_by_group[(구간, bucket)] = m["손익비"]

    print("\n===== 방향성 확인 (거래량이 많을수록 반등이 센가?) =====")
    orders = {}
    for 구간 in ["학습", "검증"]:
        pf_values = [pf_by_group.get((구간, b)) for b in BUCKET_LABELS]
        if any(v is None for v in pf_values):
            print(f"  [{구간}] 일부 거래량구간에 표본이 없어 순서를 비교할 수 없습니다.")
            orders[구간] = None
            continue

        increasing = pf_values[0] <= pf_values[1] <= pf_values[2]
        decreasing = pf_values[0] >= pf_values[1] >= pf_values[2]
        if increasing and not decreasing:
            direction = "증가(적음<보통<많음)"
        elif decreasing and not increasing:
            direction = "감소(적음>보통>많음)"
        else:
            direction = "뒤섞임(일관된 증가/감소 아님)"

        orders[구간] = direction
        print(
            f"  [{구간}] 손익비: 적음={pf_values[0]:.2f}, 보통={pf_values[1]:.2f}, "
            f"많음={pf_values[2]:.2f} -> {direction}"
        )

    print("\n===== 최종 판단 (다중비교 위험 감안 - 세 번째 필터 시도) =====")
    train_dir, val_dir = orders.get("학습"), orders.get("검증")
    train_meaningful = train_dir in ("증가(적음<보통<많음)", "감소(적음>보통>많음)")
    val_meaningful = val_dir in ("증가(적음<보통<많음)", "감소(적음>보통>많음)")

    if train_meaningful and val_meaningful and train_dir == val_dir:
        print(
            f"  학습/검증 모두 '{train_dir}' 방향이 뚜렷하게 재현됐습니다. "
            "다만 이번이 세 번째 시도이므로, 추가 종목/기간으로 한 번 더 확인한 뒤 "
            "채택 여부를 결정하는 것을 권장합니다."
        )
    else:
        print(
            "  학습/검증에서 일관되고 뚜렷한 방향성이 확인되지 않았습니다. "
            "이번 거래량 필터는 채택하지 않으며, 요청하신 전제에 따라 "
            "구름상태/피보일목 -> 이격도 크기 -> 거래량까지 세 가지 필터 모두 "
            "실패했으므로 여기서 필터 탐색을 중단하는 것을 권장합니다."
        )


def main():
    enha5 = load_merged_enha5()
    enha5 = add_volume_ratio(enha5)
    enha5, cut1, cut2 = add_volume_bucket(enha5)

    DATA_DIR.mkdir(exist_ok=True)
    enha5.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 엔하5터치 {len(enha5)}건(+거래량비율/거래량구간 컬럼)을 저장했습니다: {OUTPUT_PATH}")

    print_bucket_performance(enha5, cut1, cut2)


if __name__ == "__main__":
    main()
