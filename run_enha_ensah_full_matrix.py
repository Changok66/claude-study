# -*- coding: utf-8 -*-
"""
엔하3/4/5 진입 x 엔상3/4/5 청산의 모든 조합(3x3=9가지)을 체계적으로 비교해서
"최적 매매타점"을 찾는 스크립트.

재사용: src/reversal_angle.py의 build_entry_events()(진입 이벤트+학습/검증
구분)와 simulate_target_exit_trades()(엔상N 도달 시 청산)를 그대로 써서,
9가지 조합을 반복문으로 전부 계산한다 - 새 계산 로직은 추가하지 않았다.

각 조합마다 확인하는 것:
  1. 표본수 + 도달률(청산 조건까지 실제로 도달한 비율)
  2. 승률/손익비 (학습/검증 분리)
  3. 평균 보유봉수
  4. 실전비용(왕복 0.38%p, 이전 세션과 동일 가정) 반영 후 손익비

다중비교 방지 (요청 반영):
  - 표본 30건 미만인 조합(학습 또는 검증 어느 한쪽이라도)은 "참고용"으로
    표시하고 순위/추천에서 제외한다.
  - 학습구간 1위와 검증구간 1위가 같은 조합인지 확인한다.
  - 최종 추천은 "1등"이 아니라 "표본 충분 + 학습·검증 순위를 합쳤을 때
    가장 고르게 상위권"인 조합으로 정한다.
  - 참고(이전 세션 교훈): 조합마다 평균 보유기간이 다르므로, 손익비만 보면
    "오래 들고 있어서 유리해 보이는" 착시가 생길 수 있다. 그래서 봉당
    수익률을 연환산한 근사치도 참고용으로 같이 계산해 붙였다.

결과 파일: data/enha_ensah_full_matrix.csv (9개 조합 요약 1행씩, 매트릭스로
피벗하기 쉬운 "긴 형태(long format)"로 저장)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ma50_backtest import evaluate_trades
from src.ma50_data import KR_STOCK_CODES, fetch_kr_daily_bars
from src.reversal_angle import add_all_indicators, build_entry_events, simulate_target_exit_trades

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "enha_ensah_full_matrix.csv"

ENTRY_BANDS = (3, 4, 5)
EXIT_BANDS = (3, 4, 5)
MIN_SAMPLE_FOR_CONCLUSION = 30  # 요청에서 지정한 기준

# 이전 세션(엔하5/엔하4 검증)과 동일한 실전비용 가정.
BUY_FEE_PCT = 0.015
SELL_FEE_PCT = 0.165
SLIPPAGE_PCT_PER_SIDE = 0.1
ROUND_TRIP_COST_PCT = BUY_FEE_PCT + SELL_FEE_PCT + 2 * SLIPPAGE_PCT_PER_SIDE

warnings = []


def collect_all_trades():
    """9종목 x 9개 조합(엔하3/4/5 진입 x 엔상3/4/5 청산)의 거래를 전부 모은다."""
    all_frames = []

    for label in KR_STOCK_CODES:
        ticker, raw = fetch_kr_daily_bars(label)
        if raw.empty:
            warnings.append(f"[{label}] 데이터를 받아오지 못했습니다.")
            continue

        df = add_all_indicators(raw)

        # 종목 1개당 진입 이벤트는 밴드별로 딱 한 번씩만 계산해서 재사용한다
        # (청산 밴드 3개에 대해 매번 새로 계산할 필요 없음).
        entry_events_by_band = {
            entry_band: build_entry_events(df, entry_band) for entry_band in ENTRY_BANDS
        }

        for entry_band, events in entry_events_by_band.items():
            if events.empty:
                continue
            for exit_band in EXIT_BANDS:
                trades = simulate_target_exit_trades(df, events, exit_band)
                trades["구간"] = events["구간"].values
                trades.insert(0, "티커", ticker)
                trades.insert(0, "종목명", label)
                trades.insert(0, "청산밴드", f"엔상{exit_band}")
                trades.insert(0, "진입밴드", f"엔하{entry_band}")
                all_frames.append(trades)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def add_cost_adjusted_pnl(df):
    df = df.copy()
    df["손익률_비용반영(%)"] = df["손익률(%)"] - ROUND_TRIP_COST_PCT
    return df


def build_summary(df):
    """조합(진입밴드 x 청산밴드)별로 표본수/도달률/승률/손익비/보유기간을 요약한다."""
    rows = []

    for (entry_band, exit_band), g in df.groupby(["진입밴드", "청산밴드"], sort=False):
        n = len(g)
        도달건수 = int((g["청산사유"] != "미도달(데이터종료)").sum())
        avg_hold = g["보유봉수"].mean()
        avg_pnl = g["손익률(%)"].mean()

        row = {
            "진입밴드": entry_band,
            "청산밴드": exit_band,
            "표본수": n,
            "도달률(%)": 도달건수 / n * 100,
            "평균보유봉수": avg_hold,
            # 참고용: 보유기간이 조합마다 크게 달라서, 손익비만으로는 "오래
            # 들고 있어서 유리해 보이는" 착시가 생길 수 있다 (이전 세션에서
            # 확인한 함정) - 봉당수익률x250(연환산 근사)을 같이 남겨둔다.
            "연환산근사(%)": (avg_pnl / avg_hold * 250) if avg_hold > 0 else np.nan,
        }

        for 구간 in ["학습", "검증"]:
            sub = g[g["구간"] == 구간]
            n_sub = len(sub)
            row[f"표본수_{구간}"] = n_sub

            if n_sub == 0:
                row[f"승률_{구간}(%)"] = np.nan
                row[f"손익비_{구간}"] = np.nan
                row[f"손익비_비용반영_{구간}"] = np.nan
                continue

            m = evaluate_trades(sub)
            cost_sub = sub.drop(columns=["손익률(%)"]).rename(columns={"손익률_비용반영(%)": "손익률(%)"})
            m_cost = evaluate_trades(cost_sub)

            row[f"승률_{구간}(%)"] = m["승률(%)"]
            row[f"손익비_{구간}"] = m["손익비"]
            row[f"손익비_비용반영_{구간}"] = m_cost["손익비"]

        row["참고용_표본부족"] = (
            row.get("표본수_학습", 0) < MIN_SAMPLE_FOR_CONCLUSION
            or row.get("표본수_검증", 0) < MIN_SAMPLE_FOR_CONCLUSION
        )

        rows.append(row)

    return pd.DataFrame(rows)


def print_grid(summary, value_col, title):
    print(f"\n----- {title} -----")
    pivot = summary.pivot(index="진입밴드", columns="청산밴드", values=value_col)
    pivot = pivot.reindex(
        index=[f"엔하{b}" for b in ENTRY_BANDS],
        columns=[f"엔상{b}" for b in EXIT_BANDS],
    )
    print(pivot.round(2).to_string())


def analyze_and_recommend(summary):
    print("\n===== 학습 1위 vs 검증 1위 비교 =====")
    top_train = summary.sort_values("손익비_학습", ascending=False).iloc[0]
    top_val = summary.sort_values("손익비_검증", ascending=False).iloc[0]
    print(
        f"학습 1위: {top_train['진입밴드']}진입-{top_train['청산밴드']}청산 "
        f"(손익비 {top_train['손익비_학습']:.2f})"
    )
    print(
        f"검증 1위: {top_val['진입밴드']}진입-{top_val['청산밴드']}청산 "
        f"(손익비 {top_val['손익비_검증']:.2f})"
    )
    if (top_train["진입밴드"], top_train["청산밴드"]) == (top_val["진입밴드"], top_val["청산밴드"]):
        print("-> 학습/검증 1위가 같은 조합입니다.")
    else:
        print("-> 학습/검증 1위가 서로 다릅니다 - 단순 '1등'을 그대로 채택하면 과최적화 위험이 있습니다.")

    print("\n===== 전체 경향 확인 (진입/청산 각각의 평균 손익비) =====")
    for label, col in [("진입밴드", "진입밴드"), ("청산밴드", "청산밴드")]:
        grp = summary.groupby(col)[["손익비_학습", "손익비_검증"]].mean()
        grp = grp.reindex([f"엔{'하' if col == '진입밴드' else '상'}{b}" for b in (3, 4, 5)])
        print(f"\n{label}별 평균 손익비 (다른 축은 평균 처리):")
        print(grp.round(2).to_string())

    print(f"\n===== 최종 추천 (표본 {MIN_SAMPLE_FOR_CONCLUSION}건 이상, 학습·검증 순위 고르게 상위권) =====")
    reliable = summary[~summary["참고용_표본부족"]].copy()
    if reliable.empty:
        print("  표본 조건을 만족하는 조합이 없습니다.")
        return

    reliable["학습순위"] = reliable["손익비_학습"].rank(ascending=False)
    reliable["검증순위"] = reliable["손익비_검증"].rank(ascending=False)
    reliable["순위합"] = reliable["학습순위"] + reliable["검증순위"]
    reliable = reliable.sort_values("순위합")

    print(
        reliable[
            ["진입밴드", "청산밴드", "표본수", "손익비_학습", "손익비_검증", "학습순위", "검증순위", "순위합"]
        ].to_string(index=False)
    )

    best = reliable.iloc[0]
    print(
        f"\n-> 추천: {best['진입밴드']}진입 + {best['청산밴드']}청산 "
        f"(학습손익비 {best['손익비_학습']:.2f}, 검증손익비 {best['손익비_검증']:.2f}, "
        f"표본 {best['표본수']}건, 학습/검증 순위 {int(best['학습순위'])}/{int(best['검증순위'])}). "
        "단일 항목 1등이 아니라 학습·검증 순위를 합쳐 가장 고르게 상위권인 조합이라, "
        "어느 한쪽 구간에만 우연히 잘 맞았을 위험이 상대적으로 적습니다."
    )
    print("  (표본부족으로 제외된 조합이 있다면 위 표에 포함되지 않았으니 참고)")


def main():
    combined = collect_all_trades()
    if combined.empty:
        print("[경고] 거래가 하나도 발생하지 않았습니다.")
        return

    combined = add_cost_adjusted_pnl(combined)
    summary = build_summary(combined)

    DATA_DIR.mkdir(exist_ok=True)
    summary_sorted = summary.sort_values(["진입밴드", "청산밴드"]).reset_index(drop=True)
    summary_sorted.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(
        f"[INFO] 9개 조합 요약(1조합=1행)을 저장했습니다: {OUTPUT_PATH} "
        f"(원본 거래 {len(combined)}건 기반)"
    )

    print_grid(summary, "표본수", "표본수 매트릭스")
    print_grid(summary, "도달률(%)", "도달률(%) 매트릭스")
    print_grid(summary, "평균보유봉수", "평균보유봉수 매트릭스")
    print_grid(summary, "손익비_학습", "손익비(학습) 매트릭스")
    print_grid(summary, "손익비_검증", "손익비(검증) 매트릭스")
    print_grid(summary, "손익비_비용반영_학습", "손익비(비용반영, 학습) 매트릭스")
    print_grid(summary, "손익비_비용반영_검증", "손익비(비용반영, 검증) 매트릭스")
    print_grid(summary, "연환산근사(%)", "연환산근사(%) 매트릭스 [참고용]")

    analyze_and_recommend(summary)

    if warnings:
        print("\n===== 경고 =====")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
