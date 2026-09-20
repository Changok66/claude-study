# -*- coding: utf-8 -*-
"""
하락장 국면에서 60분봉으로 "저점 매수 - MA50 근처 매도"를 반복하는 롤링매매
전략의 신호/매매/국면판정 함수 모음이다.

매수(60분봉 기준, 아래 4가지 중 하나라도 성립하면 매수):
  1. 피보4가 국소저점 찍고 상방턴
  2. 피보5가 국소저점 찍고 상방턴
  3. 엔하4 터치 + 반전성공(반전각도가 실제로 위로 꺾임)
  4. 엔하5 터치 + 반전성공

매도: 오늘 고가가 엔상1(50봉 이동평균) 이상으로 올라오고("근처 접근"을
      이 프로젝트가 계속 써온 "터치" 관례로 구체화), 같은 날 종가가
      하방턴(국소고점 찍고 하락 전환)하면 매도.

주의(파라미터 스케일): add_all_indicators의 50/20/10 등은 "봉 개수" 기준이라
60분봉에 적용하면 "엔상1(50봉)"은 50시간(약 8~9거래일)이지 50일이 아니다.
"""

import numpy as np
import pandas as pd

from src.ma50_metrics import calc_deviation_and_gap
from src.reversal_angle import find_band_touch_events
from src.triple_indicator_confluence import _turned_up


def _turned_down(series):
    """_turned_up()의 정반대 - 국소 최고점을 찍고 하락 전환한 봉을 True로 표시한다."""
    오늘변화 = series.diff()
    어제변화 = 오늘변화.shift(1)
    return (오늘변화 < 0) & (어제변화 >= 0)


def add_buy_sell_signals(df):
    """
    df(add_all_indicators를 거친 60분봉 - 엔상1/엔하4/엔하5/피보4/피보5 필요)에
    매수 조건 4종(각각 컬럼) + 통합 매수신호 + 매도신호 컬럼을 추가한다.
    """
    df = df.copy()

    df["매수_피보4턴"] = _turned_up(df["피보4"]).fillna(False)
    df["매수_피보5턴"] = _turned_up(df["피보5"]).fillna(False)

    for band_no in (4, 5):
        col = f"매수_엔하{band_no}턴"
        events = find_band_touch_events(df, band_no=band_no)
        if events.empty:
            df[col] = False
            continue
        success_dates = set(
            events.loc[
                (events["밴드종류"] == f"엔하{band_no}터치") & events["반전성공여부"],
                "터치시각",
            ]
        )
        df[col] = df.index.isin(success_dates)

    df["매수신호"] = df[["매수_피보4턴", "매수_피보5턴", "매수_엔하4턴", "매수_엔하5턴"]].any(axis=1)

    엔상1_터치 = df["High"] >= df["엔상1"]
    하방턴 = _turned_down(df["Close"]).fillna(False)
    df["매도신호"] = 엔상1_터치 & 하방턴

    return df


BUY_REASON_COLS = ["매수_피보4턴", "매수_피보5턴", "매수_엔하4턴", "매수_엔하5턴"]


def simulate_rolling_trades(df):
    """
    add_buy_sell_signals()를 거친 df를 받아서, 무포지션일 때만 매수신호를,
    보유 중일 때만 매도신호를 보는 롤링(교차) 매매를 시뮬레이션한다.
    진입가/청산가는 그날(그 봉) 종가. 구간 끝까지 매도신호가 없으면 마지막
    봉 종가로 강제청산("미도달(구간종료)").
    """
    close = df["Close"]
    n = len(df)
    buy_sig = df["매수신호"].to_numpy()
    sell_sig = df["매도신호"].to_numpy()

    trades = []
    position = None

    for i in range(n):
        if position is None:
            if buy_sig[i]:
                reasons = [c.replace("매수_", "") for c in BUY_REASON_COLS if df[c].iloc[i]]
                position = {
                    "entry_pos": i,
                    "entry_price": close.iloc[i],
                    "매수사유": "+".join(reasons),
                }
            continue

        if i <= position["entry_pos"]:
            continue

        is_last = i == n - 1
        if sell_sig[i] or is_last:
            entry_pos = position["entry_pos"]
            entry_price = position["entry_price"]
            exit_price = close.iloc[i]
            trades.append({
                "매수사유": position["매수사유"],
                "진입일시": df.index[entry_pos],
                "진입가": entry_price,
                "청산일시": df.index[i],
                "청산가": exit_price,
                "보유봉수": i - entry_pos,
                "청산사유": "매도신호" if sell_sig[i] else "미도달(구간종료)",
                "손익률(%)": (exit_price - entry_price) / entry_price * 100,
            })
            position = None

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).sort_values("진입일시").reset_index(drop=True)


def find_bear_market_periods(daily_df, ma_window=10, threshold=0.0, min_run_days=20):
    """
    일봉 이격도(calc_deviation_and_gap, src/ma50_metrics.py)의 ma_window일
    이동평균이 threshold 미만인 날이 min_run_days일 이상 연속되는 구간들을
    "하락장 구간"으로 찾아 (시작일, 종료일) 튜플 리스트로 반환한다.

    9종목 확장 시 국면을 자동 판정하기 위한 함수 - 이번 1차(삼성전자 단독)
    실행에서는 사용자가 이미 지정한 구간이 실제로 이 기준과 겹치는지
    확인하는 참고용으로만 쓴다.
    """
    df = calc_deviation_and_gap(daily_df)
    smoothed = df["이격도"].rolling(ma_window, min_periods=ma_window).mean()
    is_bear = smoothed < threshold

    group_id = (is_bear != is_bear.shift()).cumsum()
    periods = []
    for _, g in df.groupby(group_id):
        first_idx = g.index[0]
        if bool(is_bear.loc[first_idx]) and len(g) >= min_run_days:
            periods.append((g.index[0], g.index[-1]))
    return periods


def is_true_decline(price_series, min_decline_pct=-5.0):
    """
    find_bear_market_periods()가 찾은 후보 구간(이격도 기반)은 "가격이 MA50보다
    낮다"는 것만 보장할 뿐, "그 구간 동안 가격이 실제로 떨어졌다"는 것까지는
    보장하지 않는다(2026-09-18 세션에서 HD현대중공업 등 여러 후보 구간이
    실제로는 상승 구간이었던 것을 9종목 확장 검증 중 발견함 - MA50 자체가
    하락 중이면, 가격이 반등해도 한동안 이격도가 계속 음수로 남을 수 있음).

    이 함수는 실제로 "그 구간의 시작가 대비 종가"가 min_decline_pct%(기본
    -5%) 이상 하락했는지만 판정한다. 주의: 반드시 **실제로 백테스트에 쓸
    가격 구간(예: 60분봉 데이터와 교집합해서 자른 뒤의 구간)**에 대해 호출해야
    한다 - 자르기 전의 후보 구간 전체로 판정하면, 교집합으로 잘려나간 부분의
    등락까지 섞여서 실제 백테스트 구간과 다른 결과가 나올 수 있다.
    """
    start_price = price_series.iloc[0]
    end_price = price_series.iloc[-1]
    decline_pct = (end_price / start_price - 1) * 100
    return decline_pct <= min_decline_pct, decline_pct
