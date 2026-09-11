# -*- coding: utf-8 -*-
"""
국내 보유종목 9개(일봉) - MA50 파동 시퀀스 분석 실행 스크립트.

이 파일은 src/ 안의 함수들을 순서대로 불러와서 실행하고, 결과를 data/ 폴더에
CSV 4개로 저장하는 역할만 한다. (계산 로직 자체는 src/ma50_data.py,
src/ma50_metrics.py, src/ma50_wave.py, src/ma50_backtest.py에 있다)

나스닥 선물(NQ=F)은 이번 분석에서 제외한다 (사용자 요청으로 나중으로 미룸).
기존 NQ 포함 분석 스크립트(run_ma50_analysis.py)는 그대로 둔다.

생성되는 결과 파일 (data/ 폴더):
  - ma50_wave_sequence_domestic.csv : 파동 이벤트 상세 표 (요청 1~3번 핵심 원자료)
  - ma50_wave_count_distribution.csv : 종목별/전체 파동개수 분포 (요청 3번 보조)
  - ma50_wave_decay_test.csv : 파동순번 간 감쇠비율 + 부호검정, 학습/검증 구간별 (요청4)
  - ma50_wave_backtest.csv : 파동순번별 학습/검증 백테스트 성과, 종목별+9종목 합산 (요청5)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.ma50_metrics import calc_deviation_and_gap
from src.ma50_wave import (
    build_sequence_summary,
    build_wave_events,
    decay_sign_test,
    tag_waves,
    wave_count_distribution,
)
from src.ma50_backtest import (
    evaluate_trades,
    search_best_threshold_by_wave,
    simulate_by_wave_order,
    split_train_test,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

THRESHOLD_CANDIDATES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
WAVE_ORDERS = (1, 2, 3, "4+")
MIN_TRADES_FOR_CONCLUSION = 10  # 이보다 거래수가 적으면 "표본부족"으로 표시

warnings = []


def analyze_one_stock(label, df):
    """
    종목 1개(label, 원본 일봉 df)를 받아서:
      - 파동 이벤트 표 (학습/검증 구분 포함)
      - 시퀀스 요약 / 파동개수 분포
      - 파동순번별 학습/검증 거래 목록 (evaluate 전, 풀링을 위해 원본 유지)
    을 반환한다.
    """
    df = calc_deviation_and_gap(df)
    tagged = tag_waves(df)

    wave_events = build_wave_events(tagged)
    if wave_events.empty:
        warnings.append(f"[{label}] 파동 이벤트가 하나도 발견되지 않았습니다.")
        return None

    # 시간순 70%/30% 분할 기준일을 정한다. (기존 회귀분석과 동일한 분할 방식 재사용)
    train_part, test_part = split_train_test(tagged, train_ratio=0.7)
    split_time = train_part.index[-1] if len(train_part) > 0 else tagged.index[-1]

    wave_events = wave_events.copy()
    wave_events["구분"] = np.where(wave_events["파동시각"] <= split_time, "학습", "검증")
    wave_events.insert(0, "종목", label)

    seq_summary = build_sequence_summary(wave_events)

    dist = wave_count_distribution(seq_summary)
    if not dist.empty:
        dist.insert(0, "종목", label)

    # 감쇠 검정은 학습구간/검증구간을 각각 따로 수행해서, 검증구간에서도
    # 같은 패턴(감쇠)이 유지되는지 확인한다.
    decay_rows = []
    for gubun in ["학습", "검증"]:
        subset = wave_events[wave_events["구분"] == gubun]
        d = decay_sign_test(subset)
        if not d.empty:
            d.insert(0, "구분", gubun)
            d.insert(0, "종목", label)
            decay_rows.append(d)

    # 파동순번별 백테스트: 학습구간에서 최적 임계값을 찾고 검증구간에 그대로 적용.
    backtest_rows = []
    pooled_trades = []

    for wave_order in WAVE_ORDERS:
        best_threshold, _ = search_best_threshold_by_wave(
            train_part, wave_order, thresholds=THRESHOLD_CANDIDATES
        )

        train_trades = simulate_by_wave_order(train_part, best_threshold, wave_order)
        test_trades = simulate_by_wave_order(test_part, best_threshold, wave_order)

        train_metrics = evaluate_trades(train_trades)
        test_metrics = evaluate_trades(test_trades)

        insufficient = (
            train_metrics["거래수"] < MIN_TRADES_FOR_CONCLUSION
            or test_metrics["거래수"] < MIN_TRADES_FOR_CONCLUSION
        )

        backtest_rows.append(
            {
                "종목": label,
                "파동순번": wave_order,
                "선택임계값(%)": best_threshold,
                "학습_거래수": train_metrics["거래수"],
                "학습_승률(%)": train_metrics["승률(%)"],
                "학습_손익비": train_metrics["손익비"],
                "검증_거래수": test_metrics["거래수"],
                "검증_승률(%)": test_metrics["승률(%)"],
                "검증_손익비": test_metrics["손익비"],
                "표본부족": insufficient,
            }
        )

        # 9종목을 합산한 결과도 낼 수 있도록, 원본 거래 목록에 태그를 붙여 모아둔다.
        if not train_trades.empty:
            train_trades = train_trades.copy()
            train_trades["파동순번"] = str(wave_order)
            train_trades["구분"] = "학습"
            pooled_trades.append(train_trades)
        if not test_trades.empty:
            test_trades = test_trades.copy()
            test_trades["파동순번"] = str(wave_order)
            test_trades["구분"] = "검증"
            pooled_trades.append(test_trades)

    return {
        "wave_events": wave_events,
        "dist": dist,
        "decay_rows": decay_rows,
        "backtest_rows": backtest_rows,
        "pooled_trades": pooled_trades,
        "seq_count": len(seq_summary),
    }


def main():
    all_wave_events = []
    all_dist = []
    all_decay = []
    all_backtest_rows = []
    all_pooled_trades = []
    total_seq_count = 0

    stock_names = list(KR_STOCK_CODES)
    for i, name in enumerate(stock_names, start=1):
        print(f"[{i}/{len(stock_names)}] {name} 일봉 데이터 수집 및 분석 중...")
        symbol, kr_df = fetch_kr_daily_bars(name)

        if symbol is None:
            warnings.append(f"[{name}] 야후 파이낸스 티커를 찾지 못했습니다 (.KS/.KQ 둘 다 실패).")
            continue

        if len(kr_df) < 250:
            warnings.append(f"[{name}({symbol})] 데이터가 {len(kr_df)}행으로 짧습니다 (약 1년 미만). 결과 신뢰도에 유의.")

        label = f"{name}({symbol})"
        result = analyze_one_stock(label, kr_df)
        if result is None:
            continue

        all_wave_events.append(result["wave_events"])
        if not result["dist"].empty:
            all_dist.append(result["dist"])
        all_decay.extend(result["decay_rows"])
        all_backtest_rows.extend(result["backtest_rows"])
        all_pooled_trades.extend(result["pooled_trades"])
        total_seq_count += result["seq_count"]

    # ---- 9종목 전체 시퀀스 표본 수 확인 (요청사항) ----
    print(f"\n9종목 전체 시퀀스 수 합계: {total_seq_count}개")
    if total_seq_count < 30:
        warnings.append(f"9종목 전체 시퀀스 수가 {total_seq_count}개로 30개 미만입니다. 통계 결과의 신뢰도에 주의하세요.")

    # ---- 9종목 합산(풀링) 백테스트 결과 추가 ----
    if all_pooled_trades:
        pooled = pd.concat(all_pooled_trades, ignore_index=True)
        for wave_order in WAVE_ORDERS:
            wo = str(wave_order)
            train_pool = pooled[(pooled["파동순번"] == wo) & (pooled["구분"] == "학습")]
            test_pool = pooled[(pooled["파동순번"] == wo) & (pooled["구분"] == "검증")]

            train_metrics = evaluate_trades(train_pool)
            test_metrics = evaluate_trades(test_pool)
            insufficient = (
                train_metrics["거래수"] < MIN_TRADES_FOR_CONCLUSION
                or test_metrics["거래수"] < MIN_TRADES_FOR_CONCLUSION
            )

            all_backtest_rows.append(
                {
                    "종목": "전체(9종목 합산, 종목별 최적임계값 사용)",
                    "파동순번": wave_order,
                    "선택임계값(%)": np.nan,
                    "학습_거래수": train_metrics["거래수"],
                    "학습_승률(%)": train_metrics["승률(%)"],
                    "학습_손익비": train_metrics["손익비"],
                    "검증_거래수": test_metrics["거래수"],
                    "검증_승률(%)": test_metrics["승률(%)"],
                    "검증_손익비": test_metrics["손익비"],
                    "표본부족": insufficient,
                }
            )

    # ---- 결과 저장 ----
    DATA_DIR.mkdir(exist_ok=True)

    if all_wave_events:
        result = pd.concat(all_wave_events, ignore_index=True)
        result.to_csv(DATA_DIR / "ma50_wave_sequence_domestic.csv", index=False, encoding="utf-8-sig")
        print(f"저장 완료: data/ma50_wave_sequence_domestic.csv ({len(result)}행)")
    else:
        warnings.append("파동 이벤트가 전혀 생성되지 않아 ma50_wave_sequence_domestic.csv를 저장하지 못했습니다.")

    if all_dist:
        result = pd.concat(all_dist, ignore_index=True)
        result.to_csv(DATA_DIR / "ma50_wave_count_distribution.csv", index=False, encoding="utf-8-sig")
        print(f"저장 완료: data/ma50_wave_count_distribution.csv ({len(result)}행)")

    if all_decay:
        result = pd.concat(all_decay, ignore_index=True)
        result.to_csv(DATA_DIR / "ma50_wave_decay_test.csv", index=False, encoding="utf-8-sig")
        print(f"저장 완료: data/ma50_wave_decay_test.csv ({len(result)}행)")

    if all_backtest_rows:
        result = pd.DataFrame(all_backtest_rows)
        result.to_csv(DATA_DIR / "ma50_wave_backtest.csv", index=False, encoding="utf-8-sig")
        print(f"저장 완료: data/ma50_wave_backtest.csv ({len(result)}행)")

    # ---- 경고/문제 상황 요약 출력 ----
    if warnings:
        print("\n[알림] 아래 사항을 확인해주세요:")
        for w in warnings:
            print(f" - {w}")
    else:
        print("\n모든 종목이 문제 없이 처리되었습니다.")


if __name__ == "__main__":
    main()
