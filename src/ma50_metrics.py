# -*- coding: utf-8 -*-
"""
MA50 이격도 계산, 극값(local extrema) 탐지, 구간별/체제별 통계 함수 모음이다.

핵심 아이디어 요약
------------------
1. 이격도(%) = (종가-MA50)/MA50*100  -> "가격이 MA50에서 얼마나 멀어졌는가"
2. MA간격(%) = (MA20-MA50)/MA50*100  -> "단기추세(MA20)와 중기추세(MA50)가
   얼마나 벌어져 있는가" (0에 가까우면 횡보, 많이 벌어지면 추세가 강한 상태로 본다)
3. 이격도의 부호가 바뀌는 지점(=MA50과 종가가 교차하는 지점)을 기준으로 구간을
   나누고, 그 구간 안의 최댓값/최솟값을 "극값"으로 정의한다.
   (이렇게 하면 자잘한 노이즈까지 극값으로 잡히는 것을 피할 수 있다)
"""

import numpy as np
import pandas as pd

# 이격도 통계를 낼 때 쓸 구간 경계(%) 와 각 구간의 이름표.
DEVIATION_BIN_EDGES = [0, 0.5, 1, 1.5, 2, 2.5, 3, np.inf]
DEVIATION_BIN_LABELS = [
    "0~0.5%", "0.5~1%", "1~1.5%", "1.5~2%", "2~2.5%", "2.5~3%", "3%+",
]

# MA20-MA50 간격을 "좁음/중간/넓음"으로 나누는 기준값(%).
NARROW_GAP_THRESHOLD = 0.3   # 이 값 미만이면 "좁음"(횡보 국면으로 추정)
WIDE_GAP_THRESHOLD = 1.0     # 이 값 이상이면 "넓음"(추세 국면으로 추정)


def calc_deviation_and_gap(df, price_col="Close"):
    """
    원본 가격 데이터프레임(df, "Close" 컬럼 필요)에 아래 컬럼들을 추가해서 반환한다.
      - MA20   : 최근 20개 봉 종가의 이동평균
      - MA50   : 최근 50개 봉 종가의 이동평균
      - 이격도  : (종가-MA50)/MA50*100
      - MA간격  : (MA20-MA50)/MA50*100  (양수면 MA20이 MA50 위, 음수면 아래)
    """
    # 원본을 바꾸지 않기 위해 복사본을 만든다. (다른 곳에서도 이 패턴을 쓰고 있다)
    df = df.copy()

    # rolling(n) : 최근 n개 값을 하나의 "창(window)"으로 묶어서 계산하게 해주는 기능.
    # min_periods=n : 처음 n-1개 구간처럼 창이 아직 다 안 채워졌을 때는
    #                 (부정확한 값 대신) NaN으로 비워둔다.
    df["MA20"] = df[price_col].rolling(window=20, min_periods=20).mean()
    df["MA50"] = df[price_col].rolling(window=50, min_periods=50).mean()

    df["이격도"] = (df[price_col] - df["MA50"]) / df["MA50"] * 100
    df["MA간격"] = (df["MA20"] - df["MA50"]) / df["MA50"] * 100

    return df


def classify_regime(gap_abs, narrow_th=NARROW_GAP_THRESHOLD, wide_th=WIDE_GAP_THRESHOLD):
    """MA간격의 절대값 하나를 받아서 "좁음"/"중간"/"넓음" 중 하나로 분류한다."""
    if gap_abs < narrow_th:
        return "좁음(횡보 추정)"
    elif gap_abs >= wide_th:
        return "넓음(추세 추정)"
    else:
        return "중간"


def find_extrema_events(df, price_col="Close"):
    """
    calc_deviation_and_gap()을 거친 데이터프레임을 받아서,
    "이격도 극값 이벤트"를 한 줄씩 담은 데이터프레임을 반환한다.

    각 줄에는 다음 정보가 들어간다:
      극값시각, 방향(고점/저점), 이격도(%), MA간격(%), 체제(좁음/중간/넓음),
      극값가격, 복귀시각, 복귀봉수, 복귀가격, 가격변동폭, 복귀여부
    """
    # MA20/MA50이 아직 계산 안 된(워밍업 구간의) 행은 분석에서 제외한다.
    valid = df.dropna(subset=["이격도", "MA간격"]).copy()
    if valid.empty:
        return pd.DataFrame()

    # 이격도가 양수면 1, 음수(또는 정확히 0)면 -1로 표시한 "부호 시리즈"를 만든다.
    sign = pd.Series(
        np.where(valid["이격도"] >= 0, 1, -1), index=valid.index
    )

    # sign이 바로 앞 행과 달라지는(부호가 바뀌는) 지점마다 새로운 그룹 번호를 매긴다.
    # cumsum()으로 "다르면 1, 같으면 0"을 누적해서 더하면 같은 부호가 이어지는
    # 구간끼리 같은 그룹 번호를 갖게 된다.
    group_id = (sign != sign.shift()).cumsum()

    groups = list(valid.groupby(group_id))
    records = []

    for i, (_, g) in enumerate(groups):
        seg_sign = sign.loc[g.index[0]]

        if seg_sign > 0:
            # 양의 구간 -> 그 구간에서 이격도가 가장 큰 지점 = 고점(양의 극값)
            extremum_idx = g["이격도"].idxmax()
            direction = "양의극값(고점)"
        else:
            # 음의 구간 -> 그 구간에서 이격도가 가장 작은(가장 음수인) 지점 = 저점
            extremum_idx = g["이격도"].idxmin()
            direction = "음의극값(저점)"

        extremum_pos = valid.index.get_loc(extremum_idx)
        deviation_val = valid.loc[extremum_idx, "이격도"]
        gap_val = valid.loc[extremum_idx, "MA간격"]
        regime = classify_regime(abs(gap_val))
        price_at_extremum = valid.loc[extremum_idx, price_col]

        # 다음 그룹이 존재한다면, 그 그룹의 첫 봉이 곧 "부호가 바뀐 시점"
        # = MA50으로 복귀(재교차)한 시점이다.
        if i + 1 < len(groups):
            next_g = groups[i + 1][1]
            return_idx = next_g.index[0]
            return_pos = valid.index.get_loc(return_idx)
            return_bars = return_pos - extremum_pos
            price_at_return = valid.loc[return_idx, price_col]
            price_move = price_at_return - price_at_extremum
            reverted = True
        else:
            # 마지막 그룹은 데이터가 끝날 때까지 아직 복귀하지 않은 상태
            # (이런 경우를 "미복귀/censored"라고 부르기로 한다)
            return_idx = pd.NaT
            return_bars = np.nan
            price_at_return = np.nan
            price_move = np.nan
            reverted = False

        records.append(
            {
                "극값시각": extremum_idx,
                "방향": direction,
                "이격도(%)": deviation_val,
                "MA간격(%)": gap_val,
                "체제": regime,
                "극값가격": price_at_extremum,
                "복귀시각": return_idx,
                "복귀봉수": return_bars,
                "복귀가격": price_at_return,
                "가격변동폭": price_move,
                "복귀여부": reverted,
            }
        )

    return pd.DataFrame(records)


def bucket_deviation_stats(extrema_df, n_bar_thresholds=(5, 10, 20, 50)):
    """
    극값 이벤트 표를 이격도 크기 구간(0~0.5%, 0.5~1%, ...)별로 묶어서
    표본 수 / N봉 이내 복귀확률 / 평균·중앙값 복귀봉수 / 평균 가격변동폭을 계산한다.
    """
    if extrema_df.empty:
        return pd.DataFrame()

    df = extrema_df.copy()
    df["절대이격도"] = df["이격도(%)"].abs()

    # pd.cut : 숫자를 구간(bin)으로 나눠서 각 값이 어느 구간에 속하는지 표시해준다.
    # right=False : "0.5%는 0.5~1% 구간에 포함"처럼 왼쪽 경계 포함, 오른쪽 경계 미포함.
    df["구간"] = pd.cut(
        df["절대이격도"], bins=DEVIATION_BIN_EDGES, labels=DEVIATION_BIN_LABELS, right=False
    )

    rows = []
    for label, g in df.groupby("구간", observed=True):
        if len(g) == 0:
            continue

        row = {"구간": label, "표본수": len(g)}

        for n in n_bar_thresholds:
            # "복귀했고(reverted) & 복귀까지 걸린 봉수가 n 이하"인 비율(%)
            success = g["복귀여부"] & (g["복귀봉수"] <= n)
            row[f"{n}봉이내복귀확률(%)"] = success.mean() * 100

        reverted_g = g[g["복귀여부"]]
        row["평균복귀봉수"] = reverted_g["복귀봉수"].mean()
        row["중앙값복귀봉수"] = reverted_g["복귀봉수"].median()
        row["평균가격변동폭"] = reverted_g["가격변동폭"].abs().mean()

        rows.append(row)

    return pd.DataFrame(rows)


def compare_regime_groups(extrema_df, n_bar_threshold=20):
    """
    MA20-MA50 간격 기준으로 나눈 "체제"(좁음/중간/넓음)별로
    복귀확률/복귀기간을 비교하는 표를 만든다.

    가설: "좁음(횡보)일 때 발생한 이격은 복귀 확률이 더 높고,
          넓음(추세)일 때 발생한 이격은 복귀가 잘 안 된다."
    이 표만 보면 그 가설이 실제 데이터에서도 성립하는지 바로 비교할 수 있다.
    """
    if extrema_df.empty:
        return pd.DataFrame()

    rows = []
    for regime, g in extrema_df.groupby("체제"):
        reverted_g = g[g["복귀여부"]]
        success = g["복귀여부"] & (g["복귀봉수"] <= n_bar_threshold)

        rows.append(
            {
                "체제": regime,
                "표본수": len(g),
                f"{n_bar_threshold}봉이내복귀확률(%)": success.mean() * 100,
                "평균복귀봉수": reverted_g["복귀봉수"].mean(),
                "중앙값복귀봉수": reverted_g["복귀봉수"].median(),
                "평균가격변동폭": reverted_g["가격변동폭"].abs().mean(),
            }
        )

    # 보기 좋게 좁음 -> 중간 -> 넓음 순서로 정렬한다.
    order = {"좁음(횡보 추정)": 0, "중간": 1, "넓음(추세 추정)": 2}
    result = pd.DataFrame(rows)
    result["_순서"] = result["체제"].map(order)
    result = result.sort_values("_순서").drop(columns="_순서").reset_index(drop=True)

    return result
