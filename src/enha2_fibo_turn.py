# -*- coding: utf-8 -*-
"""
엔하2(가장 안쪽 밴드) 터치 + 피보4/피보5 국소 반등 전환("피보턴")이 근접해서
겹치는 경우만 걸러낸 새 매수 신호를 만드는 함수 모음이다.

지표 계산(엔상/엔하, 피보4/5) 자체는 이미 src/reversal_angle.py에 구현돼
있고, "국소 반등 전환" 판정과 "신호 간 거리로 근접 여부 걸러내기"는 직전
세션(src/triple_indicator_confluence.py)에서 이미 만든 함수를 그대로
재사용한다. 이 파일은 그 둘을 엔하2+피보턴이라는 조합에 맞게 이어붙이는
역할만 한다.
"""

import numpy as np

from src.reversal_angle import find_band_touch_events
from src.triple_indicator_confluence import _nearest_distance, _turned_up


def add_fibo_turn_signal(df):
    """
    df(add_all_indicators를 거쳐 피보4/피보5가 있는 일봉)에 "피보턴" 컬럼을
    추가한다. 피보4 또는 피보5 중 하나라도 "어제까지 하락하다가 오늘 상승
    전환"했으면 True (직전 세션에서 쓴 _turned_up과 동일한 정의 - 오늘값이
    어제값보다 크면서, 어제값은 그제값보다 크지 않았던 날).
    """
    df = df.copy()
    df["피보턴"] = (_turned_up(df["피보4"]) | _turned_up(df["피보5"])).fillna(False)
    return df


def find_enha2_fibo_turn_events(df, search_window):
    """
    엔하2 터치 이벤트를 뽑고, 각 이벤트에서 가장 가까운 피보턴까지의
    거리(거래일 수, 부호 있음)를 계산해서 "피보턴까지거리(일)" 컬럼을 붙여
    반환한다. search_window보다 먼 피보턴은 NaN으로 처리한다
    (src.triple_indicator_confluence._nearest_distance와 동일한 규칙).

    아직 "근접 여부"로 거르지 않은 상태를 그대로 반환한다 - 호출하는 쪽에서
    1단계(필터 전 전체 표본수)를 먼저 보여준 다음, 이 거리 컬럼을 기준으로
    최종 윈도우를 정해서 걸러내도록 하기 위함이다.
    """
    events = find_band_touch_events(df, band_no=2)
    if events.empty:
        return events

    events = events[events["밴드종류"] == "엔하2터치"].copy().reset_index(drop=True)
    if events.empty:
        return events

    엔하2_위치 = [df.index.get_loc(ts) for ts in events["터치시각"]]
    피보턴_위치 = np.where(df["피보턴"].to_numpy())[0]

    events["피보턴까지거리(일)"] = _nearest_distance(엔하2_위치, 피보턴_위치, search_window)
    return events
