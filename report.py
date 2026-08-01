# -*- coding: utf-8 -*-
"""
data/sales.csv를 읽어 '월별' 그리고 '월별 x 지역별' 매출을 계산하고,
결과를 reports 폴더에 CSV로, 월별 추이는 images 폴더에 그래프로 저장한다.

지역별 총 매출 계산은 이미 analyze.py에 만들어져 있으므로
새로 만들지 않고 그대로 가져다 쓴다 (같은 로직을 두 번 만들지 않기 위함).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# analyze.py에 이미 만들어져 있는 함수를 가져와서 재사용한다.
# (같은 폴더의 .py 파일에서 함수를 import 해서 쓰는 것 = 이미 만든 코드를 재사용하는 것)
from analyze import calc_region_totals, load_sales

# 그래프에서 한글이 깨지지 않도록 폰트 설정 (analyze.py와 동일하게 설정)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = os.path.join("data", "sales.csv")
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "monthly_region_sales.csv")
IMAGE_PATH = os.path.join("images", "monthly_sales_trend.png")

# pivot_table로 표를 만들면 지역 컬럼이 가나다순(대구, 부산, 서울, 인천)으로 정렬되어
# 표를 읽을 때 헷갈리기 쉽다. 그래서 익숙한 순서로 고정해서 보여준다.
REGION_ORDER = ["서울", "부산", "대구", "인천"]


def add_month_column(df):
    # 원본 데이터프레임을 바꾸지 않기 위해 복사본을 만든다.
    df = df.copy()

    # '날짜' 컬럼은 지금 "2025-01-01" 같은 문자열이다.
    # pd.to_datetime()으로 진짜 날짜 타입으로 바꿔야 연/월을 꺼낼 수 있다.
    df["날짜"] = pd.to_datetime(df["날짜"])

    # .dt : 날짜 타입 컬럼에서 연/월/일 등을 꺼낼 때 쓰는 기능(접근자)이다.
    # strftime("%Y-%m")은 날짜를 "2025-01" 같은 "연-월" 문자열로 바꿔준다.
    df["연월"] = df["날짜"].dt.strftime("%Y-%m")

    return df


def calc_monthly_totals(df):
    # 행별 매출(수량 x 단가)을 구한 뒤, 연월별로 합산한다.
    df = df.copy()
    df["매출"] = df["수량"] * df["단가"]

    # groupby("연월") : 같은 연월끼리 묶는다.
    # ["매출"].sum() : 묶인 그룹마다 매출을 더한다.
    # sort_index() : 연월 문자열("2025-01" 등)을 오래된 순서대로 정렬한다.
    return df.groupby("연월")["매출"].sum().sort_index()


def calc_monthly_region_pivot(df):
    # 행별 매출(수량 x 단가)을 구한다.
    df = df.copy()
    df["매출"] = df["수량"] * df["단가"]

    # pivot_table : 표를 "연월"을 행(index), "지역"을 열(columns)로 재배치하면서
    # 겹치는 값(같은 연월 x 같은 지역)은 aggfunc="sum"으로 모두 더한다.
    pivot = pd.pivot_table(
        df, index="연월", columns="지역", values="매출", aggfunc="sum"
    )

    # 연월 순서대로 정렬해서 보기 좋게 만든다.
    pivot = pivot.sort_index()

    # 지역 컬럼 순서를 가나다순 대신 REGION_ORDER(서울, 부산, 대구, 인천)로 고정한다.
    # reindex(columns=...) : 지정한 순서대로 컬럼을 다시 배열한다.
    return pivot.reindex(columns=REGION_ORDER)


def save_report_csv(pivot, output_path):
    # reports 폴더가 없으면 새로 만든다 (이미 있으면 그냥 넘어간다).
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # encoding="utf-8-sig" : 엑셀에서 한글 CSV를 열 때 깨지지 않도록 하는 인코딩이다.
    pivot.to_csv(output_path, encoding="utf-8-sig")


def plot_monthly_totals(monthly_totals, output_path):
    # images 폴더가 없으면 새로 만든다.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(monthly_totals.index, monthly_totals.values, color="steelblue")
    ax.set_title("월별 총 매출 추이")
    ax.set_xlabel("연월")
    ax.set_ylabel("총 매출(원)")
    ax.ticklabel_format(style="plain", axis="y")  # y축 숫자를 지수 표기 없이 표시
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # 연월 라벨이 겹치지 않도록 기울임

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    df = load_sales(DATA_PATH)

    # 지역별 총 매출은 analyze.py의 함수를 그대로 재사용한다.
    region_totals = calc_region_totals(df)

    # 월별 집계를 위해 '연월' 컬럼을 추가한 데이터프레임을 만든다.
    df_with_month = add_month_column(df)
    monthly_totals = calc_monthly_totals(df_with_month)
    pivot = calc_monthly_region_pivot(df_with_month)

    save_report_csv(pivot, REPORT_PATH)
    plot_monthly_totals(monthly_totals, IMAGE_PATH)

    print("=== 지역별 총 매출 ===")
    print(region_totals)
    print()

    print("=== 월별 총 매출 ===")
    print(monthly_totals)
    print()

    print("=== 월별 x 지역별 매출 ===")
    print(pivot)
    print()

    print(f"{REPORT_PATH} 저장 완료")
    print(f"{IMAGE_PATH} 저장 완료")


if __name__ == "__main__":
    main()
