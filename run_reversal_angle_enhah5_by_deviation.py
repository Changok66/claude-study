# -*- coding: utf-8 -*-
"""
엔하5 롱 신호(151건)에 "이격도 크기"로 3분위 필터를 걸어서 손익비가
개선되는지 확인하는 스크립트.

이격도 정의(요청 그대로): 터치 시점 종가와 MA50(Period=50) 사이의 % 이격도.
src/reversal_angle.py의 "엔상1" 컬럼이 곧 MA50이고, 이미
data/reversal_angle_analysis.csv에 "이격도(%)" = (종가-엔상1)/엔상1*100 으로
계산되어 있으므로 다시 만들지 않고 그대로 가져다 쓴다. "이격도 크기"는 이
값의 절대값이다 (엔하5 터치는 항상 이격도가 음수이므로, 크기 비교는 절대값
기준이어야 "얼마나 더 많이 벌어졌었는가"를 뜻하게 된다).

방법 (요청 그대로):
  1. 이격도 크기를 3분위(작음/중간/큼)로 나눈다. 분위수 경계는 "학습구간"
     데이터만으로 정하고, 그 경계를 학습/검증 양쪽에 동일하게 적용한다.
     (검증구간 데이터를 미리 들여다보고 경계를 정하면 미래 정보로 규칙을
     짜는 편향이 생긴다 - 이 프로젝트가 이전 나스닥 분석 때도 지킨 "학습구간
     에서만 기준을 정한다" 원칙을 그대로 따른다)
  2. 이격도구간(작음/중간/큼) x 학습/검증 별로 승률/손익비를 비교한다.
  3. "이격도가 클수록 반등이 센지" 방향성이 학습/검증 양쪽에서 같은 순서로
     나타나는지 확인한다.

표본 부족 기준: 한 칸(이격도구간 x 학습/검증)이라도 20건 미만이면 "참고용"
으로만 표시하고 확정 결론을 내리지 않는다 (요청사항).

결과 파일: data/reversal_angle_enhah5_by_deviation.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_backtest import evaluate_trades
from src.ma50_wave import _sign_test_pvalue

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ANALYSIS_PATH = DATA_DIR / "reversal_angle_analysis.csv"
BACKTEST_PATH = DATA_DIR / "reversal_angle_backtest.csv"
OUTPUT_PATH = DATA_DIR / "reversal_angle_enhah5_by_deviation.csv"

MIN_SAMPLE_FOR_CONCLUSION = 20  # 요청에서 지정한 최소 표본 기준
JOIN_KEYS = ["종목명", "티커", "터치시각", "밴드종류"]
BUCKET_LABELS = ["작음", "중간", "큼"]


def load_merged_enha5():
    """두 결과 CSV를 합쳐서 엔하5터치 행만 골라 반환한다."""
    analysis = pd.read_csv(ANALYSIS_PATH)
    backtest = pd.read_csv(BACKTEST_PATH)

    analysis_slim = analysis[JOIN_KEYS + ["이격도(%)"]]
    merged = backtest.merge(analysis_slim, on=JOIN_KEYS, how="left")
    enha5 = merged[merged["밴드종류"] == "엔하5터치"].copy()
    return enha5


def add_deviation_bucket(df):
    """
    이격도크기(=|이격도(%)|) 컬럼을 만들고, "학습구간" 값만으로 3분위 경계
    (33.3%, 66.7% 지점)를 구해서 전체(학습+검증)에 동일하게 적용한
    "이격도구간"(작음/중간/큼) 컬럼을 추가한다.

    반환값: (구간 컬럼이 추가된 df, 경계값1, 경계값2)
    """
    df = df.copy()
    df["이격도크기"] = df["이격도(%)"].abs()

    train_values = df.loc[df["구간"] == "학습", "이격도크기"]

    # np.percentile(값들, [33.33, 66.67]) : 학습구간 값들을 정확히 3등분하는
    # 두 경계값을 구해준다. 이렇게 하면 "작음/중간/큼" 표본 수가 학습구간
    # 안에서는 거의 똑같이 맞춰진다.
    cut1, cut2 = np.percentile(train_values, [100 / 3, 200 / 3])

    def classify(v):
        if v <= cut1:
            return "작음"
        elif v <= cut2:
            return "중간"
        else:
            return "큼"

    df["이격도구간"] = df["이격도크기"].apply(classify)
    return df, cut1, cut2


def print_bucket_performance(df, cut1, cut2):
    """이격도구간 x 학습/검증 별로 표본수/승률/손익비/유의성을 출력한다."""
    print(
        f"\n[INFO] 학습구간 기준 3분위 경계 -> 작음: 이격도크기<={cut1:.2f}%, "
        f"중간: {cut1:.2f}%~{cut2:.2f}%, 큼: {cut2:.2f}% 초과"
    )

    # (구간, 버킷) -> 손익비. 나중에 방향성(작음<중간<큼 등) 비교에 쓴다.
    pf_by_group = {}

    print("\n===== 이격도구간 x 학습/검증 별 성과 =====")
    for 구간 in ["학습", "검증"]:
        for bucket in BUCKET_LABELS:
            g = df[(df["구간"] == 구간) & (df["이격도구간"] == bucket)]
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

    print("\n===== 방향성 확인 (이격도가 클수록 반등이 센가?) =====")
    orders = {}
    for 구간 in ["학습", "검증"]:
        pf_values = [pf_by_group.get((구간, b)) for b in BUCKET_LABELS]
        if any(v is None for v in pf_values):
            print(f"  [{구간}] 일부 이격도구간에 표본이 없어 순서를 비교할 수 없습니다.")
            orders[구간] = None
            continue

        increasing = pf_values[0] <= pf_values[1] <= pf_values[2]
        decreasing = pf_values[0] >= pf_values[1] >= pf_values[2]
        if increasing and not decreasing:
            direction = "증가(작음<중간<큼) - 이격도 클수록 반등이 셈"
        elif decreasing and not increasing:
            direction = "감소(작음>중간>큼) - 이격도 클수록 오히려 반등이 약함"
        else:
            direction = "뒤섞임(일관된 증가/감소 아님)"

        orders[구간] = direction
        print(
            f"  [{구간}] 손익비: 작음={pf_values[0]:.2f}, 중간={pf_values[1]:.2f}, "
            f"큼={pf_values[2]:.2f} -> {direction}"
        )

    print("\n===== 학습/검증 방향 일치 여부 (최종 확인) =====")
    train_dir, val_dir = orders.get("학습"), orders.get("검증")
    if train_dir is None or val_dir is None:
        print("  일부 구간을 비교할 수 없어 일치 여부를 판단할 수 없습니다.")
    elif train_dir == val_dir:
        print(f"  학습/검증 방향이 일치합니다 ({train_dir}) -> 재현 가능한 패턴으로 볼 수 있습니다.")
    else:
        print(
            f"  학습/검증 방향이 다릅니다 (학습: {train_dir} / 검증: {val_dir}) "
            "-> 우연일 가능성이 있으므로 이격도 크기 필터를 확정 규칙으로 채택하지 않습니다."
        )


def main():
    enha5 = load_merged_enha5()
    enha5, cut1, cut2 = add_deviation_bucket(enha5)

    DATA_DIR.mkdir(exist_ok=True)
    enha5.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 엔하5터치 {len(enha5)}건(+이격도구간 컬럼)을 저장했습니다: {OUTPUT_PATH}")

    print_bucket_performance(enha5, cut1, cut2)


if __name__ == "__main__":
    main()
