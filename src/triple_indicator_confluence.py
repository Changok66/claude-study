# -*- coding: utf-8 -*-
"""
"엔상하 + 선3일목 + 피보일목" 세 지표가 같은 시점 근처에 동시에 신호를 낼 때
매매 효과(승률/손익비)가 더 좋아지는지 검증하기 위한 함수 모음이다.

세 지표 자체의 계산 공식(엔상/엔하, 선1/선2/구름상태, 피보4/피보5)은 이미
src/reversal_angle.py에 구현되어 있으므로 여기서는 그 결과 컬럼 위에
"오늘이 신호 발생일인가?"를 판정하는 로직과, "신호들이 서로 며칠 이내에
같이 떴는가"를 계산하는 로직만 추가한다.

이 파일을 쓰려면 먼저 df에 다음 함수들을 순서대로 적용해야 한다:
  add_envelope_bands(df) -> add_span_10_20_50(df) -> add_fibo_5_8_13(df)
(run_triple_indicator_confluence.py에서 이미 이 순서로 호출한다)
"""

import numpy as np
import pandas as pd


def _turned_up(series):
    """
    시계열 하나를 받아서 "어제까지는 하락/보합이었는데 오늘 처음 상승으로
    바뀐 날"을 True로 표시하는 불리언 Series를 반환한다.

    판정 방법:
      오늘변화 = 오늘값 - 어제값
      어제변화 = 어제값 - 그제값
      "오늘변화 > 0 (오늘은 올랐다)" 그리고 "어제변화 <= 0 (어제까지는
      안 올랐다)"이 둘 다 참이면 "오늘 막 상승 전환했다"고 판정한다.

    맨 앞 1~2행처럼 어제/그제 값이 없는 구간은 비교 자체가 안 되므로
    (NaN이 섞여 조건이 False로 계산됨) 자동으로 신호에서 제외된다.
    """
    오늘변화 = series.diff()          # series.diff() = 오늘값 - 어제값
    어제변화 = 오늘변화.shift(1)       # 한 칸 뒤로 밀면 "어제 기준의 오늘변화" = 어제변화
    return (오늘변화 > 0) & (어제변화 <= 0)


def add_confluence_signals(df):
    """
    df(add_envelope_bands/add_span_10_20_50/add_fibo_5_8_13을 이미 거친 일봉)에
    3개의 신호 컬럼을 새로 추가한다. 정의는 세션 계획 문서에서 확정한 그대로다.

    신호1_엔상하 : 그날 저가(Low)가 엔하4 또는 엔하5 이하로 내려간 날
                   (매수 신호 - 이번 분석에서 매매를 실행하는 기준점)
    신호2_선3일목 : 엔상1(=MA50), 선2, 종가(Close) 세 값이 "동시에" 상승 전환한 날
    신호3_피보일목 : 피보4가 피보5를 상향 교차(골든크로스)한 날 그리고
                     같은 날 종가가 구름(선1/선2 중 더 높은 값) 위로
                     돌파한 날, 이 둘이 같은 날 동시에 일어난 날
    """
    df = df.copy()

    # ---- 신호1_엔상하: 엔하4/엔하5 중 하나라도 저가가 닿으면 신호 ----
    df["신호1_엔상하"] = ((df["Low"] <= df["엔하4"]) | (df["Low"] <= df["엔하5"])).fillna(False)

    # ---- 신호2_선3일목: MA50/선2/종가가 다같이 상승 전환 ----
    ma50_상승전환 = _turned_up(df["엔상1"])   # 엔상1 = MA(종가,50) = "50일선"
    선2_상승전환 = _turned_up(df["선2"])
    종가_상승전환 = _turned_up(df["Close"])
    df["신호2_선3일목"] = (ma50_상승전환 & 선2_상승전환 & 종가_상승전환).fillna(False)

    # ---- 신호3_피보일목: 피보4/5 골든크로스 + 구름 상단 돌파, 같은 날 동시 ----
    # 골든크로스: 어제는 피보4<=피보5였다가 오늘 피보4>피보5로 뒤집힌 날
    #  (backtest_nasdaq_futures_fibo_reversal.py의 골든크로스 판정과 동일한 방식)
    피보4_어제 = df["피보4"].shift(1)
    피보5_어제 = df["피보5"].shift(1)
    피보_골든크로스 = (df["피보4"] > df["피보5"]) & (피보4_어제 <= 피보5_어제)

    # 구름 상단(=선1/선2 중 더 큰 값) 돌파: 어제는 구름 상단 이하였다가
    # 오늘 종가가 구름 상단을 넘어선 날
    구름상단 = df[["선1", "선2"]].max(axis=1)
    구름상단_어제 = 구름상단.shift(1)
    종가_어제 = df["Close"].shift(1)
    구름_돌파 = (df["Close"] > 구름상단) & (종가_어제 <= 구름상단_어제)

    df["신호3_피보일목"] = (피보_골든크로스 & 구름_돌파).fillna(False)

    return df


def find_signal1_events(df, band_no_list=(4, 5)):
    """
    "신호1_엔상하"에 해당하는 이벤트(터치 시작일)를 뽑아 데이터프레임으로
    반환한다. 실제 판정 로직은 src/reversal_angle.py의 find_band_touch_events를
    그대로 재사용한다(3일 연속 터치를 1건으로 묶어주는 로직까지 동일하게
    적용하기 위함 - 새로 만들지 않고 검증된 함수를 그대로 쓴다).

    엔하4/엔하5 둘 다 이벤트를 뽑은 뒤 합치고, 같은 날짜에 둘 다 잡힌 경우
    (저가가 두 밴드를 동시에 뚫은 날)는 중복이므로 하나만 남긴다.
    """
    from src.reversal_angle import find_band_touch_events

    parts = []
    for band_no in band_no_list:
        events = find_band_touch_events(df, band_no=band_no)
        if events.empty:
            continue
        lower_label = f"엔하{band_no}터치"
        parts.append(events[events["밴드종류"] == lower_label])

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    # 같은 날짜가 여러 밴드에서 동시에 잡혔으면 하나만 남긴다(첫 번째 것 유지).
    combined = combined.drop_duplicates(subset="터치시각").sort_values("터치시각").reset_index(drop=True)
    return combined


def _nearest_distance(signal_positions, target_positions, search_window):
    """
    signal_positions(신호1이 일어난 정수 위치들의 배열) 각각에 대해,
    target_positions(신호2 또는 신호3이 일어난 정수 위치들, 오름차순 정렬된
    배열) 중 가장 가까운 위치까지의 "며칠 차이"를 구한다.

    반환값: signal_positions와 길이가 같은 리스트. 각 원소는
      - 실수(정수값이지만 NaN과 섞이면 pandas가 자동으로 float 컬럼을 만들어
        주도록 float로 반환한다. 신호2/3이 앞이면 음수, 뒤면 양수, 같은 날이면 0)
      - search_window보다 멀거나 target이 아예 없으면 np.nan
        (파이썬 None을 쓰면 나중에 컬럼 dtype이 object가 돼서 평균/중앙값 같은
        집계 함수가 에러를 낼 수 있어서, "값 없음"은 항상 np.nan으로 통일한다)

    np.searchsorted로 "삽입할 위치"를 찾은 뒤, 그 바로 앞/뒤 후보만 비교하면
    되므로 전체를 다 뒤지지 않고도 가장 가까운 값을 빠르게 찾을 수 있다.
    """
    if len(target_positions) == 0:
        return [np.nan] * len(signal_positions)

    target_positions = np.asarray(target_positions)
    distances = []

    for pos in signal_positions:
        insert_at = np.searchsorted(target_positions, pos)
        candidates = []
        if insert_at < len(target_positions):
            candidates.append(target_positions[insert_at])
        if insert_at > 0:
            candidates.append(target_positions[insert_at - 1])

        # 후보 중 pos와 가장 가까운(절댓값 차이가 가장 작은) 것을 고른다.
        best = min(candidates, key=lambda t: abs(t - pos))
        diff = float(best - pos)

        distances.append(diff if abs(diff) <= search_window else np.nan)

    return distances


def compute_signal_distances(df, signal1_events, search_window=10):
    """
    signal1_events(find_signal1_events의 결과, "터치시각" 컬럼 포함)의 각
    이벤트에 대해, 같은 종목 df 안에서 신호2/신호3까지의 거리(일수)를 계산해
    "신호2까지거리(일)", "신호3까지거리(일)" 컬럼을 추가해서 반환한다.

    search_window: 이 범위(거래일 수) 밖에 있는 신호는 "너무 멀어서 관련
    없다"고 보고 None(거리 없음) 처리한다. 2단계(분포 확인)에서는 넉넉하게
    ±10일로 잡아서 실제 분포를 먼저 살펴보고, 3단계 동시성 판정에서는 그
    분포를 보고 정한 더 좁은 윈도우를 쓴다.
    """
    signal1_events = signal1_events.copy()

    # 날짜(Timestamp)를 df 안에서의 정수 위치(몇 번째 봉인지)로 바꾼다.
    signal1_positions = [df.index.get_loc(ts) for ts in signal1_events["터치시각"]]
    signal2_positions = np.where(df["신호2_선3일목"].to_numpy())[0]
    signal3_positions = np.where(df["신호3_피보일목"].to_numpy())[0]

    signal1_events["신호2까지거리(일)"] = _nearest_distance(
        signal1_positions, signal2_positions, search_window
    )
    signal1_events["신호3까지거리(일)"] = _nearest_distance(
        signal1_positions, signal3_positions, search_window
    )

    return signal1_events


def decide_confluence_window(all_distances, min_window=1, max_window=10, fallback=3):
    """
    2단계에서 모은 "신호2까지거리(일)"/"신호3까지거리(일)" 값들(None 제외,
    9종목 전체 합산)을 보고, 3단계에서 쓸 "동시성 판정 윈도우"를 정한다.

    방법: 거리의 절댓값의 중앙값(median)을 반올림해서 쓴다 - 실제로 신호들이
    서로 며칠 이내에 같이 뜨는 경향이 있는지를 데이터로 확인한 뒤 그 값을
    그대로 기준으로 삼는 것("동시에 떴다"의 기준을 감으로 정하지 않기 위함).
    값이 없거나(신호2/3이 아예 같이 뜬 적이 없음) 이상한 값이면 fallback으로
    정한 기본값(3거래일)을 대신 쓴다. min_window~max_window 범위로 잘라서
    너무 좁거나(0일) 너무 넓은(수십일) 윈도우가 나오는 것을 막는다.
    """
    valid = [abs(d) for d in all_distances if not pd.isna(d)]
    if not valid:
        return fallback

    median_distance = int(round(float(np.median(valid))))
    return max(min_window, min(max_window, median_distance))


def add_confluence_group(signal1_events_with_distance, window):
    """
    compute_signal_distances()를 거친 signal1_events_with_distance에
    "동시신호개수"(1/2/3), "신호2존재여부", "신호3존재여부" 컬럼을 추가한다.

    동시신호개수 = 1(신호1 자기 자신) + (신호2가 window 이내에 있으면 1) +
                  (신호3가 window 이내에 있으면 1)
    """
    df = signal1_events_with_distance.copy()

    def _within(distance):
        return (not pd.isna(distance)) and abs(distance) <= window

    df["신호2존재여부"] = df["신호2까지거리(일)"].apply(_within)
    df["신호3존재여부"] = df["신호3까지거리(일)"].apply(_within)
    df["동시신호개수"] = 1 + df["신호2존재여부"].astype(int) + df["신호3존재여부"].astype(int)

    return df
