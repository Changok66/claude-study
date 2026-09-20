# -*- coding: utf-8 -*-
"""
엔하2+피보턴 거래(src/enha2_fibo_turn.py)에 추세선4/5(src/fibo_trendline_breakout.py)
방향을 "참조용 필터/등급"으로 붙이는 함수 모음이다. 추세선 자체를 진입
트리거로 쓰지 않고, 이미 발생한 엔하2+피보턴 신호를 거르거나 등급 매기는
용도로만 쓴다.
"""


def attach_trendline_direction(df, trades):
    """
    trades("터치시각" 컬럼 필요, 진입일)의 각 행에 대해 그날의 추세선4/추세선5
    방향("상승"=오늘 값>어제 값, "하락"=그 외)을 붙이고, 그 방향을 그대로
    "강한신호"(상승)/"약한신호"(하락) 등급으로도 붙인다.

    df는 add_trend_lines()를 거친 것이어야 한다(추세선4/추세선5 컬럼 필요).
    """
    trades = trades.copy()

    dir4 = (df["추세선4"] > df["추세선4"].shift(1)).map({True: "상승", False: "하락"})
    dir5 = (df["추세선5"] > df["추세선5"].shift(1)).map({True: "상승", False: "하락"})

    trades["추세선방향(수식4)"] = trades["터치시각"].map(dir4)
    trades["추세선방향(수식5)"] = trades["터치시각"].map(dir5)
    trades["강약등급(수식4)"] = trades["추세선방향(수식4)"].map({"상승": "강한신호", "하락": "약한신호"})
    trades["강약등급(수식5)"] = trades["추세선방향(수식5)"].map({"상승": "강한신호", "하락": "약한신호"})

    return trades
