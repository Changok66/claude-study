# -*- coding: utf-8 -*-
"""
매출 데이터를 지역별/월별/제품별 등 다양한 기준으로 집계(계산)하는 함수 모음이다.
"""

import math

import pandas as pd

# pivot_table로 표를 만들면 지역 컬럼이 가나다순(대구, 부산, 서울, 인천)으로 정렬되어
# 표를 읽을 때 헷갈리기 쉽다. 그래서 익숙한 순서로 고정해서 보여준다.
#
# data/make_data.py(데이터 생성)와 report.ipynb(그래프 색상 매핑)에서도 똑같은
# 지역/제품 이름 목록이 필요하다. 목록을 여러 파일에 각각 적어두면 나중에 지역이나
# 제품이 바뀔 때 한쪽만 고치고 다른 쪽을 깜빡하기 쉬우므로, 여기 이 두 목록을
# "단일 기준"으로 두고 다른 파일에서는 이 목록을 가져다 쓴다.
REGION_ORDER = ["서울", "부산", "대구", "인천"]
PRODUCT_ORDER = ["노트북", "마우스", "키보드", "모니터"]


def _add_revenue_column(df):
    # 아래 여러 계산 함수가 공통으로 필요로 하는 "행별 매출(수량 x 단가)" 컬럼을
    # 추가해주는 내부용 도우미 함수다. (직접 df["매출"] = ... 을 반복해서 쓰지 않기 위함)
    # 원본 데이터프레임을 바꾸지 않기 위해 복사본을 만든다.
    df = df.copy()
    df["매출"] = df["수량"] * df["단가"]
    return df


def calc_region_totals(df):
    # 행별 매출(수량 x 단가)을 구한 뒤 지역별로 합산
    df = _add_revenue_column(df)
    return df.groupby("지역")["매출"].sum().sort_values(ascending=False)


def calc_monthly_totals(df):
    # 행별 매출(수량 x 단가)을 구한 뒤, 연월별로 합산한다.
    df = _add_revenue_column(df)

    # groupby("연월") : 같은 연월끼리 묶는다.
    # ["매출"].sum() : 묶인 그룹마다 매출을 더한다.
    # sort_index() : 연월 문자열("2025-01" 등)을 오래된 순서대로 정렬한다.
    return df.groupby("연월")["매출"].sum().sort_index()


def calc_monthly_region_pivot(df):
    # 행별 매출(수량 x 단가)을 구한다.
    df = _add_revenue_column(df)

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


def calc_product_totals(df):
    # 행별 매출(수량 x 단가)을 구한 뒤 제품별로 합산
    df = _add_revenue_column(df)

    # sort_values(ascending=False) : 매출이 큰 제품부터 내림차순으로 정렬한다.
    return df.groupby("제품")["매출"].sum().sort_values(ascending=False)


def calc_region_product_pivot(df):
    # 행별 매출(수량 x 단가)을 구한다.
    df = _add_revenue_column(df)

    # pivot_table : 표를 "지역"을 행(index), "제품"을 열(columns)로 재배치하면서
    # 겹치는 값(같은 지역 x 같은 제품)은 aggfunc="sum"으로 모두 더한다.
    pivot = pd.pivot_table(
        df, index="지역", columns="제품", values="매출", aggfunc="sum"
    )

    # 지역 행 순서를 가나다순 대신 REGION_ORDER(서울, 부산, 대구, 인천)로 고정한다.
    return pivot.reindex(index=REGION_ORDER)


def calc_region_bestseller(df):
    # 지역 x 제품 매출표는 이미 만들어져 있는 calc_region_product_pivot을 그대로 가져다 쓴다.
    # (같은 계산을 두 번 만들지 않기 위해서다)
    pivot = calc_region_product_pivot(df)

    # 어떤 지역에 매출이 전혀 없으면(그 행의 모든 제품이 NaN) idxmax/max는
    # "가장 큰 값"을 찾을 수 없어 에러를 낸다. 그런 지역은 애초에 베스트셀러를
    # 정할 수 없으므로, 계산 전에 결과에서 미리 제외한다.
    # dropna(how="all") : 행의 모든 값이 NaN인 행만 제거한다.
    pivot = pivot.dropna(how="all")

    # idxmax(axis=1) : 각 행(지역)에서 값이 가장 큰 "컬럼 이름"(제품명)을 찾아준다.
    # max(axis=1) : 각 행(지역)에서 가장 큰 값(매출액) 자체를 찾아준다.
    bestseller = pd.DataFrame(
        {
            "베스트셀러 제품": pivot.idxmax(axis=1),
            "매출": pivot.max(axis=1),
        }
    )

    return bestseller


def calc_quarterly_totals(df):
    # 행별 매출(수량 x 단가)을 구한 뒤, 분기별로 합산한다.
    # 이 함수를 쓰려면 df에 "분기" 컬럼이 미리 있어야 한다 (data.add_quarter_column으로 추가).
    df = _add_revenue_column(df)

    # groupby("분기") : 같은 분기(1~4)끼리 묶는다.
    # sort_index() : 분기 번호를 1 -> 2 -> 3 -> 4 순서로 정렬한다.
    return df.groupby("분기")["매출"].sum().sort_index()


def calc_pareto_analysis(product_totals):
    # product_totals : calc_product_totals의 결과(매출 내림차순 정렬된 제품별 매출)를 받는다.

    총_제품_수 = len(product_totals)
    총_매출 = product_totals.sum()

    # 상위 20%에 해당하는 제품 개수를 구한다.
    # 예) 제품이 4개면 4 x 0.2 = 0.8개인데, 제품은 1개 단위이므로 올림해서 최소 1개는 되게 한다.
    상위_20퍼센트_개수 = max(1, math.ceil(총_제품_수 * 0.2))

    # 매출이 가장 큰 제품부터 상위_20퍼센트_개수 만큼 뽑아 매출 합을 구한다.
    상위_제품_매출_합 = product_totals.iloc[:상위_20퍼센트_개수].sum()

    # 상위 20% 제품의 매출이 전체 매출에서 차지하는 비율(%)
    상위_20퍼센트_비율 = 상위_제품_매출_합 / 총_매출 * 100

    # 파레토 차트를 그릴 때 필요한 표를 만든다.
    # 매출(원)과 누적비율(%)은 단위가 완전히 달라서 그래프 한 축에 같이 그리면
    # 이중 축(축이 2개인 그래프)이 되어 오히려 읽기 어려워진다.
    # 그래서 "매출" 대신 "비율(%)"로 통일해, 막대(개별 비율)와 꺾은선(누적 비율)을
    # 같은 0~100% 축 하나에 그릴 수 있도록 한다.
    누적표 = pd.DataFrame(
        {
            "매출": product_totals,
            "비율(%)": product_totals / 총_매출 * 100,
            "누적비율(%)": product_totals.cumsum() / 총_매출 * 100,
        }
    )

    return {
        "상위_20퍼센트_제품_개수": 상위_20퍼센트_개수,
        "상위_20퍼센트_비율": 상위_20퍼센트_비율,
        "누적표": 누적표,
    }
