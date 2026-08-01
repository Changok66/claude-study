# -*- coding: utf-8 -*-
"""
data/sales.csv를 읽고, 분석하기 편한 형태로 다듬는(전처리) 함수 모음이다.
"""

import pandas as pd


def load_sales(path):
    # CSV를 읽어 데이터프레임으로 반환
    return pd.read_csv(path, encoding="utf-8-sig")


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


def add_quarter_column(df):
    # 원본 데이터프레임을 바꾸지 않기 위해 복사본을 만든다.
    df = df.copy()

    # '날짜' 컬럼을 문자열에서 진짜 날짜 타입으로 바꾼다. (add_month_column과 동일)
    df["날짜"] = pd.to_datetime(df["날짜"])

    # .dt.quarter : 날짜가 몇 분기(1~4)에 속하는지 숫자로 알려주는 기능이다.
    # 예) 1~3월이면 1, 4~6월이면 2, 7~9월이면 3, 10~12월이면 4
    df["분기"] = df["날짜"].dt.quarter

    return df
