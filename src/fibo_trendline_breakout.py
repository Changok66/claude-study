# -*- coding: utf-8 -*-
"""
새 추세선 2종(수식4=MA13+MA19 평균, 수식5=MA15+MA21 평균)을 만들고, 피보4
또는 피보5가 이 추세선을 아래->위로 돌파하면 매수, 위->아래로 재돌파하면
매도하는 단순 크로스오버 전략의 신호/매매 로직이다.

"피보 4 또는 5", "추세선 수식4 또는 5" 둘 다 사용자가 아직 확정하지 못한
상태라, 하나로 좁혀 짐작하지 않고 가능한 4가지 조합(피보4x수식4, 피보4x수식5,
피보5x수식4, 피보5x수식5)을 전부 계산해서 나란히 비교한다
(run_enha_ensah_full_matrix.py가 "엔하3/4/5 진입 x 엔상3/4/5 청산" 9가지를
다중비교했던 것과 같은 접근).
"""

import pandas as pd

MA_PAIRS = {
    "수식4": (13, 19),
    "수식5": (15, 21),
}

FIBO_COLS = ["피보4", "피보5"]


def add_trend_lines(df):
    """
    df(Close 컬럼 필요)에 "추세선4"(MA13+MA19 평균), "추세선5"(MA15+MA21 평균)
    컬럼을 추가한다.
    """
    df = df.copy()
    for name, (short_period, long_period) in MA_PAIRS.items():
        ma_short = df["Close"].rolling(window=short_period).mean()
        ma_long = df["Close"].rolling(window=long_period).mean()
        df[f"추세선{name[-1]}"] = (ma_short + ma_long) / 2
    return df


def find_breakout_trades(df, fibo_col, trend_col):
    """
    fibo_col(예: "피보4")이 trend_col(예: "추세선4")을 상향 돌파하면 매수,
    하향 재돌파하면 매도하는 단순 교차 전략의 거래 표를 반환한다.

    무포지션 상태에서는 "상향 돌파"만, 보유 중에는 "하향 재돌파"만 보는
    교차(alternating) 방식이다 - 이미 보유 중일 때 추가 매수 신호가 나와도
    무시한다(물타기/중복진입 없음, stock-study의 detect_fibo_crossovers +
    교차 포지션 로직과 동일한 관례).

    진입가/청산가는 이 프로젝트의 기존 관례대로 그날 종가로 체결됐다고
    가정한다. 데이터 끝까지 매도 신호가 안 나오면 마지막 봉 종가로
    강제청산("미도달(데이터종료)").
    """
    fibo = df[fibo_col]
    trend = df[trend_col]
    close = df["Close"]
    n = len(df)

    above = fibo > trend  # 오늘 피보가 추세선 위에 있는지 (NaN이면 자동 False)

    trades = []
    position = None  # 보유 중이면 {"entry_pos":, "entry_price":}

    for i in range(1, n):
        was_above = bool(above.iloc[i - 1])
        is_above = bool(above.iloc[i])

        if position is None:
            if (not was_above) and is_above:
                # 상향 돌파 = 매수
                position = {"entry_pos": i, "entry_price": close.iloc[i]}
        else:
            if was_above and (not is_above):
                # 하향 재돌파 = 매도
                entry_pos = position["entry_pos"]
                entry_price = position["entry_price"]
                exit_price = close.iloc[i]
                trades.append({
                    "진입일": df.index[entry_pos],
                    "진입가": entry_price,
                    "청산일": df.index[i],
                    "청산가": exit_price,
                    "보유봉수": i - entry_pos,
                    "청산사유": "하향재돌파",
                    "손익률(%)": (exit_price - entry_price) / entry_price * 100,
                })
                position = None

    if position is not None:
        # 데이터 끝까지 매도 신호가 안 나온 경우 마지막 봉 종가로 강제청산.
        entry_pos = position["entry_pos"]
        entry_price = position["entry_price"]
        exit_pos = n - 1
        exit_price = close.iloc[exit_pos]
        trades.append({
            "진입일": df.index[entry_pos],
            "진입가": entry_price,
            "청산일": df.index[exit_pos],
            "청산가": exit_price,
            "보유봉수": exit_pos - entry_pos,
            "청산사유": "미도달(데이터종료)",
            "손익률(%)": (exit_price - entry_price) / entry_price * 100,
        })

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades).sort_values("진입일").reset_index(drop=True)
