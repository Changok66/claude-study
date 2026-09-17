# -*- coding: utf-8 -*-
"""
엔하5 롱 신호에 선3일목/피보일목 참고 컬럼을 "필터 조건"으로 걸었을 때
손익비가 실제로 개선되는지 확인하는 스크립트.

배경: run_reversal_angle_backtest.py에서 엔하5 터치->롱 진입 규칙은 이미
괜찮았다 (학습 손익비 1.74, 검증 1.69). 그때는 구름상태(선3일목)/피보4·피보5
(피보일목)를 "참고용 컬럼"으로만 남겨뒀는데, 이번엔 이 값들을 진짜 필터로
걸어서 더 좋아지는지, 아니면 표본만 줄고 별 차이가 없는지 확인한다.

시도하는 조건 (각각 독립적으로 하나씩 확인):
  1. 구름상태 == "양운"   (하락 중 반등이 아니라, 상승추세 중 눌림목 반등만 인정)
  2. 피보4 > 피보5        (단기 피보 중간값이 장기 피보 중간값보다 위 = 골든크로스)
  3. 1과 2를 모두 만족

data/reversal_angle_analysis.csv(참고 컬럼 보유)와
data/reversal_angle_backtest.csv(실제 매매 손익 보유)를 이미 만들어뒀으므로,
지표를 다시 계산하지 않고 두 표를 (종목명,티커,터치시각,밴드종류) 기준으로
합쳐서 재사용한다.

결과 파일: data/reversal_angle_enhah5_filtered.csv
(엔하5터치 거래 전체 + 조건1/2/3 충족 여부 컬럼)
학습/검증 구간별 승률/손익비/유의성 비교는 콘솔에 출력한다.
"""

from pathlib import Path

import pandas as pd

from src.ma50_backtest import evaluate_trades
from src.ma50_wave import _sign_test_pvalue

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ANALYSIS_PATH = DATA_DIR / "reversal_angle_analysis.csv"
BACKTEST_PATH = DATA_DIR / "reversal_angle_backtest.csv"
OUTPUT_PATH = DATA_DIR / "reversal_angle_enhah5_filtered.csv"

# 요청에서 지정한 기준: 30건 미만이면 "참고용"으로만 표시하고 확정 결론은 안 냄.
# (run_reversal_angle_backtest.py의 종목별 10건 기준과는 별개 - 이번엔 필터를
#  거는 것이라 표본이 더 잘게 쪼개지므로 더 보수적인 기준을 쓴다)
MIN_SAMPLE_FOR_CONCLUSION = 30

JOIN_KEYS = ["종목명", "티커", "터치시각", "밴드종류"]


def load_merged_enha5():
    """두 결과 CSV를 읽어서 엔하5터치 행만 골라 하나로 합친 표를 반환한다."""
    analysis = pd.read_csv(ANALYSIS_PATH)
    backtest = pd.read_csv(BACKTEST_PATH)

    # analysis 쪽에서는 조인키 + 이번에 새로 필요한 참고 컬럼(구름상태/피보4/피보5)만
    # 남긴다. (반전각도(도)/반전성공여부/구간처럼 두 표에 겹치는 컬럼은 backtest
    # 쪽 값을 그대로 쓰면 되므로 여기서는 가져오지 않는다 - 중복 컬럼 방지)
    analysis_slim = analysis[JOIN_KEYS + ["구름상태", "피보4", "피보5"]]

    merged = backtest.merge(analysis_slim, on=JOIN_KEYS, how="left")
    enha5 = merged[merged["밴드종류"] == "엔하5터치"].copy()
    return enha5


def add_condition_flags(df):
    """조건1/2/3을 만족하는지를 True/False 컬럼으로 추가한다."""
    df = df.copy()
    df["조건1_양운"] = df["구름상태"] == "양운"
    df["조건2_피보골든크로스"] = df["피보4"] > df["피보5"]
    df["조건3_양운and골든크로스"] = df["조건1_양운"] & df["조건2_피보골든크로스"]
    return df


def evaluate_condition(df, condition_mask, condition_label):
    """
    조건 하나(condition_mask가 None이면 필터 없음)를 학습/검증 구간별로 나눠서
    표본수/승률/손익비/반전성공률의 통계적 유의성을 콘솔에 출력한다.
    """
    print(f"\n--- {condition_label} ---")
    subset = df[condition_mask] if condition_mask is not None else df

    for 구간 in ["학습", "검증"]:
        g = subset[subset["구간"] == 구간]
        n = len(g)

        if n == 0:
            print(f"  [{구간}] 표본 없음")
            continue

        m = evaluate_trades(g)
        k = int(g["반전성공여부"].sum())
        p_value = _sign_test_pvalue(n, k)

        표본_안내 = f"{n}건"
        if n < MIN_SAMPLE_FOR_CONCLUSION:
            표본_안내 += f" [참고용 - {MIN_SAMPLE_FOR_CONCLUSION}건 미만이라 확정 결론 보류]"

        print(
            f"  [{구간}] 표본 {표본_안내}: 승률={m['승률(%)']:.1f}%, "
            f"손익비={m['손익비']:.2f}, 반전성공률={k / n * 100:.1f}%(p={p_value:.4f})"
        )


def main():
    enha5 = load_merged_enha5()
    enha5 = add_condition_flags(enha5)

    DATA_DIR.mkdir(exist_ok=True)
    enha5.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 엔하5터치 {len(enha5)}건(+조건 플래그 컬럼 포함)을 저장했습니다: {OUTPUT_PATH}")

    print("\n===== 기준선: 필터 없음 (run_reversal_angle_backtest.py 결과와 같아야 함) =====")
    evaluate_condition(enha5, None, "필터 없음 (엔하5터치 전체)")

    print("\n===== 조건별 필터 적용 결과 =====")
    evaluate_condition(enha5, enha5["조건1_양운"], "조건1: 구름상태 = 양운")
    evaluate_condition(enha5, enha5["조건2_피보골든크로스"], "조건2: 피보4 > 피보5 (골든크로스)")
    evaluate_condition(enha5, enha5["조건3_양운and골든크로스"], "조건3: 조건1 AND 조건2")


if __name__ == "__main__":
    main()
