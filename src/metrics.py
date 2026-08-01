# -*- coding: utf-8 -*-
"""
매출 데이터를 지역별/월별/제품별 등 다양한 기준으로 집계(계산)하는 함수 모음이다.
"""

import pandas as pd

# pivot_table로 표를 만들면 지역 컬럼이 가나다순(대구, 부산, 서울, 인천)으로 정렬되어
# 표를 읽을 때 헷갈리기 쉽다. 그래서 익숙한 순서로 고정해서 보여준다.
REGION_ORDER = ["서울", "부산", "대구", "인천"]


def calc_region_totals(df):
    # 행별 매출(수량 x 단가)을 구한 뒤 지역별로 합산
    df = df.copy()
    df["매출"] = df["수량"] * df["단가"]
    return df.groupby("지역")["매출"].sum().sort_values(ascending=False)


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


def calc_product_totals(df):
    # 행별 매출(수량 x 단가)을 구한 뒤 제품별로 합산
    df = df.copy()
    df["매출"] = df["수량"] * df["단가"]

    # sort_values(ascending=False) : 매출이 큰 제품부터 내림차순으로 정렬한다.
    return df.groupby("제품")["매출"].sum().sort_values(ascending=False)


def calc_region_product_pivot(df):
    # 행별 매출(수량 x 단가)을 구한다.
    df = df.copy()
    df["매출"] = df["수량"] * df["단가"]

    # pivot_table : 표를 "지역"을 행(index), "제품"을 열(columns)로 재배치하면서
    # 겹치는 값(같은 지역 x 같은 제품)은 aggfunc="sum"으로 모두 더한다.
    pivot = pd.pivot_table(
        df, index="지역", columns="제품", values="매출", aggfunc="sum"
    )

    # 지역 행 순서를 가나다순 대신 REGION_ORDER(서울, 부산, 대구, 인천)로 고정한다.
    return pivot.reindex(index=REGION_ORDER)
