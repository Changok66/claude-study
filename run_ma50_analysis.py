# -*- coding: utf-8 -*-
"""
MA50 이격도 기반 평균회귀 패턴 분석 - 실행 스크립트.

이 파일은 src/ 안의 함수들(데이터 수집, 이격도/극값 계산, 백테스트)을 순서대로
불러와서 실제로 실행하고, 결과를 data/ 폴더에 CSV 3개로 저장하는 역할만 한다.
(계산 로직 자체는 src/ 쪽 파일들에 있고, 이 파일은 "그 로직들을 조합해서 돌리는" 파일이다)

분석 대상 (총 10개):
  1) 나스닥100 E-mini 선물(NQ=F) - 4분봉
  2) 국내 보유종목 9개 - 일봉
     (삼성전자, SK하이닉스, 두산에너빌리티, 테크윙, 아모레퍼시픽,
      HD현대중공업, HD현대일렉트릭, 에이피알, 대한광통신)

생성되는 결과 파일 (data/ 폴더):
  - ma50_deviation_multi_analysis.csv : 극값(이탈->복귀) 이벤트 상세 표
  - ma50_deviation_bucket_stats.csv   : 이격도 크기 구간별 복귀확률/복귀기간 통계
  - ma50_deviation_regime_compare.csv : MA20-MA50 간격(좁음/중간/넓음)별 비교
  - ma50_deviation_backtest.csv       : 학습구간/검증구간 백테스트 결과
                                         (필터없음 / 좁음필터 / 넓음필터 비교)
"""

from pathlib import Path

import pandas as pd

from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars, fetch_nq_4min_bars
from src.ma50_backtest import (
    evaluate_trades,
    search_best_threshold,
    simulate_threshold_strategy,
    split_train_test,
)
from src.ma50_metrics import (
    bucket_deviation_stats,
    calc_deviation_and_gap,
    compare_regime_groups,
    find_extrema_events,
)

# 이 스크립트 파일이 있는 폴더(프로젝트 루트) 기준으로 data 폴더 경로를 만든다.
# (실행 위치와 상관없이 항상 올바른 data 폴더를 가리키게 하기 위함)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 백테스트에서 시도해볼 이격도 임계값 후보들(%).
THRESHOLD_CANDIDATES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

# 하나의 CSV 안에서 어떤 실패/경고가 있었는지 마지막에 모아서 보여주기 위한 리스트.
warnings = []


def process_instrument(label, timeframe, raw_df, extrema_all, bucket_all, regime_all, backtest_all):
    """
    종목 1개를 받아서 이격도 계산 -> 극값 탐지 -> 구간별/체제별 통계 -> 백테스트까지
    전부 수행하고, 결과를 각각의 리스트(extrema_all 등)에 추가한다.
    """
    if raw_df is None or raw_df.empty or len(raw_df) < 60:
        n = 0 if raw_df is None else len(raw_df)
        warnings.append(f"[{label}] 데이터가 없거나 너무 적어({n}행) 분석을 건너뜁니다.")
        return

    # 1) 이격도 / MA간격 계산
    df = calc_deviation_and_gap(raw_df)

    # 2) 극값(이탈 -> 복귀) 이벤트 탐지
    extrema = find_extrema_events(df)
    if extrema.empty:
        warnings.append(f"[{label}] 극값이 하나도 발견되지 않았습니다.")
        return

    # 종목/타임프레임 정보를 표 맨 앞 컬럼으로 붙여서 나중에 여러 종목을 합쳐도
    # 어떤 데이터인지 구분할 수 있게 한다.
    extrema = extrema.copy()
    extrema.insert(0, "타임프레임", timeframe)
    extrema.insert(0, "종목", label)
    extrema_all.append(extrema)

    # 사용자가 요청한 대로, NQ(4분봉)의 극값 개수가 20개 미만이면 명확히 경고한다.
    if timeframe == "4분봉" and len(extrema) < 20:
        warnings.append(
            f"[{label}] 극값 개수가 {len(extrema)}개로 20개 미만입니다. "
            "표본이 적어 통계적 유의성이 부족할 수 있습니다."
        )

    # 3) 이격도 크기 구간별 통계
    bstats = bucket_deviation_stats(extrema)
    if not bstats.empty:
        bstats = bstats.copy()
        bstats.insert(0, "타임프레임", timeframe)
        bstats.insert(0, "종목", label)
        bucket_all.append(bstats)

    # 4) MA20-MA50 간격(체제) 별 비교
    rcompare = compare_regime_groups(extrema)
    if not rcompare.empty:
        rcompare = rcompare.copy()
        rcompare.insert(0, "타임프레임", timeframe)
        rcompare.insert(0, "종목", label)
        regime_all.append(rcompare)

    # 5) 백테스트: 필터없음 / 좁음필터(횡보국면만 진입) / 넓음필터(추세국면만 진입) 비교
    train_df, test_df = split_train_test(df, train_ratio=0.7)

    for filter_name, filter_value in [("필터없음", None), ("좁음필터", "좁음"), ("넓음필터", "넓음")]:
        best_threshold, _ = search_best_threshold(
            train_df, thresholds=THRESHOLD_CANDIDATES, regime_filter=filter_value
        )

        train_trades = simulate_threshold_strategy(train_df, best_threshold, regime_filter=filter_value)
        test_trades = simulate_threshold_strategy(test_df, best_threshold, regime_filter=filter_value)

        train_metrics = evaluate_trades(train_trades)
        test_metrics = evaluate_trades(test_trades)

        backtest_all.append(
            {
                "종목": label,
                "타임프레임": timeframe,
                "필터": filter_name,
                "선택임계값(%)": best_threshold,
                "학습_거래수": train_metrics["거래수"],
                "학습_승률(%)": train_metrics["승률(%)"],
                "학습_손익비": train_metrics["손익비"],
                "검증_거래수": test_metrics["거래수"],
                "검증_승률(%)": test_metrics["승률(%)"],
                "검증_손익비": test_metrics["손익비"],
            }
        )


def main():
    extrema_all = []
    bucket_all = []
    regime_all = []
    backtest_all = []

    # ---- 1) 나스닥100 선물(NQ=F) 4분봉 ----
    print("[1/10] 나스닥100 선물(NQ=F) 4분봉 데이터 수집 중...")
    nq_df = fetch_nq_4min_bars()
    process_instrument("나스닥100선물(NQ=F)", "4분봉", nq_df, extrema_all, bucket_all, regime_all, backtest_all)

    # ---- 2~10) 국내 보유종목 9개 일봉 ----
    for i, name in enumerate(KR_STOCK_CODES, start=2):
        print(f"[{i}/10] {name} 일봉 데이터 수집 중...")
        symbol, kr_df = fetch_kr_daily_bars(name)

        if symbol is None:
            warnings.append(f"[{name}] 야후 파이낸스 티커를 찾지 못했습니다 (.KS/.KQ 둘 다 실패).")
            continue

        process_instrument(f"{name}({symbol})", "일봉", kr_df, extrema_all, bucket_all, regime_all, backtest_all)

    # ---- 결과 저장 ----
    DATA_DIR.mkdir(exist_ok=True)

    if extrema_all:
        result = pd.concat(extrema_all, ignore_index=True)
        # encoding="utf-8-sig" : 엑셀에서 한글 CSV를 열어도 깨지지 않도록 하는 인코딩.
        result.to_csv(DATA_DIR / "ma50_deviation_multi_analysis.csv", index=False, encoding="utf-8-sig")
        print(f"저장 완료: data/ma50_deviation_multi_analysis.csv ({len(result)}행)")
    else:
        warnings.append("극값 이벤트가 전혀 생성되지 않아 ma50_deviation_multi_analysis.csv를 저장하지 못했습니다.")

    if bucket_all:
        result = pd.concat(bucket_all, ignore_index=True)
        result.to_csv(DATA_DIR / "ma50_deviation_bucket_stats.csv", index=False, encoding="utf-8-sig")
        print(f"저장 완료: data/ma50_deviation_bucket_stats.csv ({len(result)}행)")

    if regime_all:
        result = pd.concat(regime_all, ignore_index=True)
        result.to_csv(DATA_DIR / "ma50_deviation_regime_compare.csv", index=False, encoding="utf-8-sig")
        print(f"저장 완료: data/ma50_deviation_regime_compare.csv ({len(result)}행)")

    if backtest_all:
        result = pd.DataFrame(backtest_all)
        result.to_csv(DATA_DIR / "ma50_deviation_backtest.csv", index=False, encoding="utf-8-sig")
        print(f"저장 완료: data/ma50_deviation_backtest.csv ({len(result)}행)")

    # ---- 경고/문제 상황 요약 출력 ----
    if warnings:
        print("\n[알림] 아래 사항을 확인해주세요:")
        for w in warnings:
            print(f" - {w}")
    else:
        print("\n모든 종목이 문제 없이 처리되었습니다.")


if __name__ == "__main__":
    main()
