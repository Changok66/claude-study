# -*- coding: utf-8 -*-
"""
국내 보유종목 9개(일봉) - 엔상5/엔하5 밴드 반전각도 분석 실행 스크립트.

이 파일은 src/ 안의 함수들을 순서대로 불러와서 실행하고, 결과를 data/ 폴더에
CSV 1개로 저장하는 역할만 한다. (계산 로직 자체는 src/reversal_angle.py에 있다)

동작 순서(종목마다 반복):
  1. src/ma50_data.py로 일봉 데이터를 받는다.
  2. src/reversal_angle.py의 함수들로 엔상/엔하 밴드, 참고용 일목 지표를 계산한다.
  3. 엔상5/엔하5 터치 이벤트(터치 전후 5봉 반전각도 포함)를 찾는다.
  4. 학습구간(70%)/검증구간(30%)으로 나눠 "구분" 컬럼을 붙인다.
     (분할 기준은 src/ma50_backtest.py의 split_train_test를 그대로 재사용 -
     기존 run_ma50_wave_domestic.py와 동일한 방식: 원본 일봉을 시간순으로
     70%/30% 나눈 뒤, 그 경계 날짜를 기준으로 이벤트를 학습/검증으로 분류한다)

생성되는 결과 파일: data/reversal_angle_analysis.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.ma50_backtest import split_train_test
from src.reversal_angle import (
    add_envelope_bands,
    add_fibo_5_8_13,
    add_span_10_20_50,
    find_band_touch_events,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "reversal_angle_analysis.csv"

N_BARS = 5  # 터치 전후로 기울기를 볼 봉 개수 (요청 파라미터)

warnings = []


def analyze_one_stock(label, ticker, df):
    """
    종목 1개(label=종목명, ticker=야후파이낸스 티커, df=원본 일봉)를 받아서
    엔상5/엔하5 터치 이벤트 표(학습/검증 구분 + 종목 정보 포함)를 반환한다.
    이벤트가 하나도 없으면 None을 반환한다.
    """
    # ---- 1. 지표 계산 (전체 기간을 끊지 않고 한 번에 계산해야 MA/ATR
    #         워밍업이 검증구간 앞부분에서 끊기지 않는다) ----
    df = add_envelope_bands(df)
    df = add_span_10_20_50(df)
    df = add_fibo_5_8_13(df)

    # ---- 2. 터치 이벤트 + 반전각도 계산 ----
    events = find_band_touch_events(df, n_bars=N_BARS)
    if events.empty:
        warnings.append(f"[{label}] 엔상5/엔하5 터치 이벤트가 하나도 발견되지 않았습니다.")
        return None

    # ---- 3. 학습/검증 구간 나누기 (기존 run_ma50_wave_domestic.py와 동일한 방식) ----
    train_part, _ = split_train_test(df, train_ratio=0.7)
    split_time = train_part.index[-1] if len(train_part) > 0 else df.index[-1]

    events = events.copy()
    events["구간"] = np.where(events["터치시각"] <= split_time, "학습", "검증")
    events.insert(0, "티커", ticker)
    events.insert(0, "종목명", label)

    return events


def print_summary(all_events):
    """
    구간(학습/검증) x 밴드종류(엔상5터치/엔하5터치)별로
    이벤트 수 / 평균 반전각도 / 반전성공률을 콘솔에 출력한다.
    실행 직후 결과가 말이 되는지 눈으로 바로 확인하기 위한 용도다.
    """
    print("\n===== 구간 x 밴드종류별 요약 (9종목 합산) =====")
    grouped = all_events.groupby(["구간", "밴드종류"])

    for (구간, 밴드종류), g in grouped:
        평균반전각도 = g["반전각도(도)"].mean()
        반전성공률 = g["반전성공여부"].mean() * 100
        print(
            f"[{구간}] {밴드종류}: 이벤트 {len(g)}건, "
            f"평균 반전각도 {평균반전각도:.2f}도, 반전성공률 {반전성공률:.1f}%"
        )


def main():
    all_events = []

    for label in KR_STOCK_CODES:
        ticker, df = fetch_kr_daily_bars(label)
        if df.empty:
            warnings.append(f"[{label}] 데이터를 받아오지 못했습니다 (야후 파이낸스 조회 실패).")
            continue

        # yfinance가 주는 인덱스(시각)를 "Date" 이름의 보통 컬럼처럼 다루기 위해
        # datetime 인덱스를 그대로 두고 계산한다 (다른 ma50_*.py 함수들도 동일하게
        # datetime 인덱스를 기준으로 동작하므로 맞춰준다).
        print(f"[INFO] {label}({ticker}) 처리 중... ({len(df)}행)")
        result = analyze_one_stock(label, ticker, df)
        if result is not None:
            all_events.append(result)

    if not all_events:
        print("[경고] 9종목 전부에서 터치 이벤트가 발견되지 않았습니다. 결과 파일을 만들지 않습니다.")
        return

    combined = pd.concat(all_events, ignore_index=True)
    combined = combined.sort_values(["종목명", "터치시각"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    # encoding="utf-8-sig" : 엑셀에서 한글 컬럼명이 깨지지 않도록 BOM을 붙여 저장한다.
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[INFO] 총 {len(combined)}건의 터치 이벤트를 저장했습니다: {OUTPUT_PATH}")
    print_summary(combined)

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
