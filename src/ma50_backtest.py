# -*- coding: utf-8 -*-
"""
"이격도 극값 = 반대매매 진입신호" 전략의 백테스트(과거 데이터로 모의투자) 함수 모음이다.

중요한 설계 원칙: 룩어헤드 편향(lookahead bias) 방지
--------------------------------------------------
"극값 지점"은 사실 그 이후에 가격이 반전해야만 사후적으로 알 수 있는 지점이다.
실시간 매매에서는 "지금이 극값이다"를 미리 알 수 없다. 그래서 실전처럼 시뮬레이션
하기 위해, 진입 신호는 "이격도가 처음으로 기준치(임계값) 이상 벌어지는 순간"으로
정의한다. (미래 정보를 미리 아는 셈이 되는 오류를 피하기 위함)

전략 규칙
---------
- 롱(매수) 진입 : 이격도가 -임계값 이하로 처음 내려간 순간
- 숏(매도) 진입 : 이격도가 +임계값 이상으로 처음 올라간 순간
- 청산(매매종료) : 이격도가 0을 다시 넘어와 MA50으로 복귀했을 때
  (혹은 보유기간이 max_hold_bars를 넘으면 강제 청산)
- 한 번에 포지션(보유 중인 매매)은 하나만 유지한다 (보유 중엔 새 진입 신호 무시)
"""

import numpy as np
import pandas as pd

from src.ma50_metrics import NARROW_GAP_THRESHOLD, WIDE_GAP_THRESHOLD


def split_train_test(df, train_ratio=0.7):
    """
    시간순으로 정렬된 데이터프레임을 앞부분(학습구간)/뒷부분(검증구간)으로 나눈다.
    시계열 데이터는 미래 데이터가 과거 학습에 섞이면 안 되므로 무작위 분할이 아니라
    "앞 train_ratio 비율까지는 학습, 그 뒤는 검증"처럼 시간 순서를 지켜서 나눈다.
    """
    n = len(df)
    split_point = int(n * train_ratio)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    return train_df, test_df


def _run_trade_simulation(rows, threshold_pct, entry_allowed, max_hold_bars, price_col):
    """
    진입/청산 매매 루프의 공통 부분을 모아둔 내부용 함수.
    "지금 이 봉에서 신규 진입을 허용할지"만 entry_allowed(row) 함수로 넘겨받고,
    나머지(임계값 돌파 체크, 청산 조건, 강제청산 마무리)는 여기서 동일하게 처리한다.
    simulate_threshold_strategy(체제 필터)와 simulate_by_wave_order(파동순번 필터)가
    이 함수를 함께 사용한다.
    """
    trades = []
    position = None  # 포지션이 없으면 None, 있으면 진입정보를 담은 dict

    # iterrows()로 한 봉씩 순서대로 확인한다. (학습용 코드라 속도보다 이해하기
    # 쉬운 방식을 택했다. 데이터 규모가 커도 수천~수만 행 수준이라 문제 없다)
    for i, (idx, row) in enumerate(rows.iterrows()):
        deviation = row["이격도"]
        price = row[price_col]

        if position is None:
            # ---- 진입 여부 판단 ----
            if entry_allowed(row):
                if deviation <= -threshold_pct:
                    position = {"방향": "롱", "진입시각": idx, "진입가": price, "진입순번": i}
                elif deviation >= threshold_pct:
                    position = {"방향": "숏", "진입시각": idx, "진입가": price, "진입순번": i}
        else:
            # ---- 청산 여부 판단 ----
            hold_bars = i - position["진입순번"]
            should_exit = False

            if position["방향"] == "롱" and deviation >= 0:
                should_exit = True  # MA50까지 복귀
            elif position["방향"] == "숏" and deviation <= 0:
                should_exit = True

            if max_hold_bars is not None and hold_bars >= max_hold_bars:
                should_exit = True

            if should_exit:
                pnl_pct = (price - position["진입가"]) / position["진입가"] * 100
                if position["방향"] == "숏":
                    pnl_pct = -pnl_pct  # 숏은 가격이 내려가야 이익

                trades.append(
                    {
                        "방향": position["방향"],
                        "진입시각": position["진입시각"],
                        "진입가": position["진입가"],
                        "청산시각": idx,
                        "청산가": price,
                        "보유봉수": hold_bars,
                        "손익률(%)": pnl_pct,
                        "강제청산": False,
                    }
                )
                position = None

    # 데이터 끝까지 청산되지 않은 포지션은 마지막 가격 기준으로 강제 청산 처리한다.
    # (실제로는 청산이 안 된 상태이므로 "강제청산" 표시를 남겨 구분한다)
    if position is not None and len(rows) > 0:
        last_idx = rows.index[-1]
        last_price = rows.iloc[-1][price_col]
        pnl_pct = (last_price - position["진입가"]) / position["진입가"] * 100
        if position["방향"] == "숏":
            pnl_pct = -pnl_pct

        trades.append(
            {
                "방향": position["방향"],
                "진입시각": position["진입시각"],
                "진입가": position["진입가"],
                "청산시각": last_idx,
                "청산가": last_price,
                "보유봉수": len(rows) - 1 - position["진입순번"],
                "손익률(%)": pnl_pct,
                "강제청산": True,
            }
        )

    return pd.DataFrame(trades)


def simulate_threshold_strategy(
    df,
    threshold_pct,
    regime_filter=None,
    narrow_th=NARROW_GAP_THRESHOLD,
    wide_th=WIDE_GAP_THRESHOLD,
    max_hold_bars=None,
    price_col="Close",
):
    """
    이격도 임계값 전략을 한 번 시뮬레이션해서 개별 거래(trade) 목록을 반환한다.

    regime_filter:
      None   -> MA20-MA50 간격을 보지 않고 이격도 임계값만으로 진입 (필터 없음)
      "좁음" -> MA간격의 절대값이 narrow_th 미만일 때만 진입 허용 (횡보 국면 필터)
      "넓음" -> MA간격의 절대값이 wide_th 이상일 때만 진입 허용 (추세 국면 필터)
    """
    rows = df.dropna(subset=["이격도", "MA간격", price_col])

    def entry_allowed(row):
        gap_abs = abs(row["MA간격"])
        if regime_filter == "좁음" and gap_abs >= narrow_th:
            return False
        if regime_filter == "넓음" and gap_abs < wide_th:
            return False
        return True

    return _run_trade_simulation(rows, threshold_pct, entry_allowed, max_hold_bars, price_col)


def simulate_by_wave_order(df_tagged, threshold_pct, target_wave_order, max_hold_bars=None, price_col="Close"):
    """
    src.ma50_wave.tag_waves()로 "파동순번" 컬럼이 매겨진 데이터프레임을 받아서,
    "지금 이 봉이 target_wave_order번째 파동에 속할 때만" 진입을 허용하는
    시뮬레이션을 수행한다.

    target_wave_order : 1, 2, 3 (그 번째 파동에서만 진입) 또는 "4+" (4번째 이상
                         모든 파동에서 진입 허용, 표본이 적은 후반 파동들을 묶기 위함)
    """
    rows = df_tagged.dropna(subset=["이격도", price_col])

    def entry_allowed(row):
        wave_no = row.get("파동순번", np.nan)
        if pd.isna(wave_no):
            # "MA20까지 되돌림" 구간 등 파동에 속하지 않는 봉은 진입 대상이 아니다.
            return False
        if target_wave_order == "4+":
            return wave_no >= 4
        return int(wave_no) == target_wave_order

    return _run_trade_simulation(rows, threshold_pct, entry_allowed, max_hold_bars, price_col)


def evaluate_trades(trades_df):
    """
    거래 목록(trades_df)을 받아서 승률/손익비/거래수 등 성과지표를 계산한다.
      - 승률(%)  : 이익이 난 거래의 비율
      - 손익비   : 총이익 / |총손실|  (1보다 크면 "이익 낼 때 번 돈이 손실 낼 때
                   잃은 돈보다 많다"는 뜻. 값이 클수록 좋다)
    """
    if trades_df.empty:
        return {"거래수": 0, "승률(%)": np.nan, "손익비": np.nan}

    wins = trades_df.loc[trades_df["손익률(%)"] > 0, "손익률(%)"]
    losses = trades_df.loc[trades_df["손익률(%)"] <= 0, "손익률(%)"]

    win_rate = len(wins) / len(trades_df) * 100

    total_loss = abs(losses.sum())
    if total_loss == 0:
        # 손실이 하나도 없으면 손익비를 무한대로 둔다 (분모가 0이 되는 것을 피함).
        profit_factor = np.inf if wins.sum() > 0 else np.nan
    else:
        profit_factor = wins.sum() / total_loss

    return {"거래수": len(trades_df), "승률(%)": win_rate, "손익비": profit_factor}


def search_best_threshold(
    train_df, thresholds=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), min_trades=3, **simulate_kwargs
):
    """
    학습구간(train_df)에서 여러 임계값 후보를 각각 시뮬레이션해보고,
    (거래수가 min_trades 이상인 후보들 중에서) 손익비가 가장 좋은 임계값을 고른다.

    반환값: (선택된 최적 임계값, 모든 후보의 성과를 담은 비교표)
    """
    results = []
    for th in thresholds:
        trades = simulate_threshold_strategy(train_df, th, **simulate_kwargs)
        metrics = evaluate_trades(trades)
        metrics["임계값(%)"] = th
        results.append(metrics)

    table = pd.DataFrame(results)

    # 거래 표본이 너무 적은 임계값은 "우연히 성과가 좋아 보일 위험"이 크므로 제외한다.
    reliable = table[table["거래수"] >= min_trades]

    if reliable.empty:
        # 표본 조건을 만족하는 임계값이 하나도 없으면, 그래도 거래가 가장 많았던
        # 임계값을 골라 최소한의 결과라도 낸다.
        best_row = table.sort_values("거래수", ascending=False).iloc[0]
    else:
        best_row = reliable.sort_values("손익비", ascending=False).iloc[0]

    return best_row["임계값(%)"], table


def search_best_threshold_by_wave(
    train_df, target_wave_order, thresholds=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), min_trades=3, **simulate_kwargs
):
    """
    search_best_threshold()와 동일한 방식이되, "파동순번 == target_wave_order"인
    구간에서만 진입을 허용한 채로 임계값 후보들을 비교해서 최적값을 고른다.
    """
    results = []
    for th in thresholds:
        trades = simulate_by_wave_order(train_df, th, target_wave_order, **simulate_kwargs)
        metrics = evaluate_trades(trades)
        metrics["임계값(%)"] = th
        results.append(metrics)

    table = pd.DataFrame(results)
    reliable = table[table["거래수"] >= min_trades]

    if reliable.empty:
        best_row = table.sort_values("거래수", ascending=False).iloc[0]
    else:
        best_row = reliable.sort_values("손익비", ascending=False).iloc[0]

    return best_row["임계값(%)"], table
