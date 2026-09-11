# -*- coding: utf-8 -*-
"""
"파동 시퀀스" 탐지 함수 모음이다.

배경: 기존 src/ma50_metrics.py의 find_extrema_events()는 "MA50 이탈 시작부터
MA50 완전 복귀까지"를 통째로 1개 사이클로 보고, 그 안에서 가장 크게 벌어진
지점(극값) 딱 1개만 뽑았다. 그런데 실제로는 그 사이클 안에서

  이격 -> MA20까지만 되돌림(MA50까지는 아직 안 돌아옴) -> 다시 이격
  -> MA20 되돌림 -> 다시 이격 -> ... -> 결국 MA50 완전 복귀

처럼 여러 번(파동) 오르내리다가 마지막에야 MA50으로 돌아오는 경우가 많다.
이 파일은 그 "여러 번의 파동"을 각각 구분해서 몇 번째 파동인지 표시해준다.

용어 정리
---------
- 시퀀스(sequence) : MA50 이탈 시작 ~ MA50 완전 복귀까지의 큰 틀 (기존 극값 탐지의
  "구간"과 같은 개념).
- 파동(wave)       : 시퀀스 안에서, 가격이 MA20보다도 더 바깥쪽에 있는(=이격이
  진행 중인) 하위 구간 하나하나. 파동과 파동 사이에는 "MA20까지 되돌림" 구간이
  끼어 있다 (이 되돌림 구간 자체는 파동으로 세지 않는다).
"""

import numpy as np
import pandas as pd


def tag_waves(df, price_col="Close"):
    """
    calc_deviation_and_gap()을 거친 데이터프레임(이격도/MA20/MA50 컬럼 필요)에
    아래 3개 컬럼을 추가해서 반환한다.
      - 시퀀스ID   : 몇 번째 "MA50 이탈~복귀" 사이클인지 (정수)
      - 시퀀스방향 : "양(고점)" 또는 "음(저점)"
      - 파동순번   : 그 시퀀스 안에서 몇 번째 파동인지 (1,2,3...).
                    "MA20까지 되돌림" 구간에 속한 행은 NaN으로 남긴다.
    """
    df = df.copy()
    df["시퀀스ID"] = np.nan
    df["시퀀스방향"] = None
    df["파동순번"] = np.nan

    # MA20/MA50/이격도가 아직 계산 안 된(워밍업) 행은 제외한다.
    valid = df.dropna(subset=["이격도", "MA간격"])
    if valid.empty:
        return df

    # ---- 1단계: 시퀀스 나누기 (MA50 기준 부호가 같은 동안 = 같은 시퀀스) ----
    seq_sign = pd.Series(np.where(valid["이격도"] >= 0, 1, -1), index=valid.index)
    seq_group_id = (seq_sign != seq_sign.shift()).cumsum()

    df.loc[valid.index, "시퀀스ID"] = seq_group_id.values
    df.loc[valid.index, "시퀀스방향"] = np.where(seq_sign.values > 0, "양(고점)", "음(저점)")

    # ---- 2단계: 각 시퀀스 내부를 MA20 기준으로 다시 나눠서 "파동" 찾기 ----
    for _, seq_rows in valid.groupby(seq_group_id):
        parent_sign = seq_sign.loc[seq_rows.index[0]]

        # 이 시퀀스 내부에서, 종가가 MA20보다 위/아래인지의 부호.
        sub_sign = pd.Series(
            np.where(seq_rows[price_col] - seq_rows["MA20"] >= 0, 1, -1),
            index=seq_rows.index,
        )
        sub_group_id = (sub_sign != sub_sign.shift()).cumsum()

        wave_no = 0
        for _, sub_rows in seq_rows.groupby(sub_group_id):
            this_sub_sign = sub_sign.loc[sub_rows.index[0]]

            # 하위구간의 부호가 시퀀스 전체 부호와 "같을 때"만 진짜 파동
            # (종가가 MA20보다도 더 바깥쪽에 있어서 이격이 진행 중인 상태).
            # 부호가 반대면 "MA20까지 되돌림" 구간이므로 파동으로 세지 않는다.
            if this_sub_sign == parent_sign:
                wave_no += 1
                df.loc[sub_rows.index, "파동순번"] = wave_no

    return df


def build_wave_events(tagged_df, price_col="Close"):
    """
    tag_waves()를 거친 데이터프레임에서, 각 파동을 대표하는 지점(그 파동 안에서
    이격도가 가장 큰/작은 지점) 하나씩을 뽑아 이벤트 표로 만든다.
    """
    tagged = tagged_df.dropna(subset=["시퀀스ID"])
    if tagged.empty:
        return pd.DataFrame()

    seq_groups = list(tagged.groupby("시퀀스ID"))
    records = []

    for i, (seq_id, seq_rows) in enumerate(seq_groups):
        seq_dir = seq_rows["시퀀스방향"].iloc[0]
        seq_start = seq_rows.index[0]

        # 다음 시퀀스가 있으면, 그 시퀀스의 첫 봉이 곧 "MA50 완전 복귀" 시점이다.
        if i + 1 < len(seq_groups):
            seq_return_idx = seq_groups[i + 1][1].index[0]
            seq_reverted = True
        else:
            # 마지막 시퀀스는 데이터가 끝날 때까지 아직 복귀하지 못한 상태다.
            seq_return_idx = pd.NaT
            seq_reverted = False

        wave_rows = seq_rows.dropna(subset=["파동순번"])
        if wave_rows.empty:
            continue
        total_waves = int(wave_rows["파동순번"].max())

        for wave_no, wg in wave_rows.groupby("파동순번"):
            # 이 파동 안에서 이격도가 가장 극단적인(고점이면 최댓값, 저점이면 최솟값)
            # 지점을 그 파동의 대표 지점으로 삼는다.
            if seq_dir == "양(고점)":
                rep_idx = wg["이격도"].idxmax()
            else:
                rep_idx = wg["이격도"].idxmin()

            records.append(
                {
                    "시퀀스ID": seq_id,
                    "시퀀스방향": seq_dir,
                    "시퀀스시작시각": seq_start,
                    "시퀀스복귀시각": seq_return_idx,
                    "시퀀스복귀여부": seq_reverted,
                    "시퀀스총파동수": total_waves,
                    "파동순번": int(wave_no),
                    "파동시각": rep_idx,
                    "이격도(%)": tagged.loc[rep_idx, "이격도"],
                    "가격": tagged.loc[rep_idx, price_col],
                }
            )

    return pd.DataFrame(records)


def build_sequence_summary(wave_events):
    """파동 이벤트 표를 시퀀스 단위로 압축해서, 시퀀스 1개당 1행인 표를 만든다."""
    if wave_events.empty:
        return pd.DataFrame()

    return (
        wave_events.groupby("시퀀스ID")
        .agg(
            시퀀스방향=("시퀀스방향", "first"),
            시퀀스시작시각=("시퀀스시작시각", "first"),
            시퀀스복귀여부=("시퀀스복귀여부", "first"),
            총파동수=("시퀀스총파동수", "first"),
        )
        .reset_index()
    )


def wave_count_distribution(sequence_summary):
    """시퀀스별 파동 개수(1개/2개/3개/4개 이상)의 분포표를 만든다."""
    if sequence_summary.empty:
        return pd.DataFrame()

    # 4개 이상은 전부 "4개 이상"으로 묶는다 (clip으로 4를 상한으로 자름).
    capped = sequence_summary["총파동수"].clip(upper=4)
    labels = {1: "1개", 2: "2개", 3: "3개", 4: "4개 이상"}

    counts = capped.value_counts().sort_index()
    result = pd.DataFrame(
        {
            "파동수": [labels[int(v)] for v in counts.index],
            "시퀀스개수": counts.values,
        }
    )
    return result


def _sign_test_pvalue(n, k):
    """
    이항분포 Binomial(n, 0.5) 기준 부호검정(sign test)의 양측 p-value를 계산한다.

    아이디어: "감쇠와 무관하게 순전히 우연이라면(50:50 확률)" n번 중 k번(또는 그보다
    더 치우친 결과)이 나올 확률이 얼마나 작은지를 본다. 이 확률(p-value)이 작을수록
    "우연이 아니라 진짜 패턴일 가능성이 높다"고 해석한다. (통상 0.05 미만이면 유의미)

    math.comb(n, i) : n개 중 i개를 고르는 조합의 수. scipy 없이 표준 라이브러리
    math만으로 이항분포 확률을 직접 계산할 수 있다.
    """
    from math import comb

    if n == 0:
        return np.nan

    # 결과가 한쪽으로 치우친 정도를 재기 위해, k와 n-k 중 더 큰 쪽을 기준으로 삼는다
    # (이항분포는 대칭이라 어느 쪽으로 계산해도 같은 p-value가 나온다).
    k_extreme = max(k, n - k)

    # P(X >= k_extreme) 을 직접 합산한다.
    one_side = sum(comb(n, i) for i in range(k_extreme, n + 1)) / (2**n)

    # 양측검정이므로 2배 해준다 (반대쪽 극단도 같은 확률로 대칭이므로).
    p_value = min(1.0, one_side * 2)
    return p_value


def decay_sign_test(wave_events):
    """
    "다음 파동이 이전 파동보다 이격폭이 작아지는가(감쇠)"를 검증한다.
    같은 시퀀스 안에서 파동1 vs 파동2, 파동2 vs 파동3, ... 을 쌍으로 비교한다.
    """
    if wave_events.empty:
        return pd.DataFrame()

    events = wave_events.copy()
    events["절대이격도"] = events["이격도(%)"].abs()

    # 시퀀스ID를 행, 파동순번을 열로 하는 표로 바꾼다.
    # 이렇게 하면 같은 시퀀스의 1차/2차/3차 파동 값을 한 줄에서 바로 비교할 수 있다.
    pivot = events.pivot_table(index="시퀀스ID", columns="파동순번", values="절대이격도")

    max_wave = int(events["파동순번"].max())
    rows = []

    for cur in range(1, max_wave):
        nxt = cur + 1
        if cur not in pivot.columns or nxt not in pivot.columns:
            continue

        # 두 파동값이 "둘 다 존재하는" 시퀀스만 비교 대상이 된다.
        pair = pivot[[cur, nxt]].dropna()
        n = len(pair)
        if n == 0:
            continue

        decayed = int((pair[nxt] < pair[cur]).sum())  # 다음 파동이 더 작았던 횟수
        decay_rate = decayed / n * 100
        p_value = _sign_test_pvalue(n, decayed)

        rows.append(
            {
                "비교": f"{cur}차->{nxt}차",
                "표본수(시퀀스쌍)": n,
                "감쇠비율(%)": decay_rate,
                "부호검정_p값": p_value,
                "유의미(p<0.05)": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            }
        )

    return pd.DataFrame(rows)
