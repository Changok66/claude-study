# test_report.py
# report.py의 계산 함수(add_month_column, calc_monthly_totals, calc_monthly_region_pivot)가
# 숫자를 정확히 계산하는지 검증하는 pytest 테스트입니다.

import pandas as pd
import pytest

from src.data import add_month_column
from src.metrics import calc_monthly_region_pivot, calc_monthly_totals


@pytest.fixture
def sample_df():
    # 손으로 매출을 계산할 수 있는 작은 데이터를 직접 만듭니다.
    # 실제 sales.csv 대신 이 데이터를 쓰는 이유:
    # 결과가 맞는지 사람이 직접 검산할 수 있어야 테스트가 의미 있기 때문입니다.
    return pd.DataFrame(
        {
            "날짜": ["2025-01-01", "2025-01-02", "2025-02-01"],
            "지역": ["서울", "서울", "부산"],
            "수량": [2, 3, 1],
            "단가": [1000, 1000, 5000],
        }
    )


def test_add_month_column(sample_df):
    # add_month_column을 실행하면 '연월' 컬럼이 새로 생겨야 합니다.
    df = add_month_column(sample_df)

    # "2025-01-01" 날짜는 "2025-01"로, "2025-02-01" 날짜는 "2025-02"로 바뀌어야 합니다.
    assert list(df["연월"]) == ["2025-01", "2025-01", "2025-02"]


def test_calc_monthly_totals(sample_df):
    df = add_month_column(sample_df)
    result = calc_monthly_totals(df)

    # 2025-01 매출 = (2개 x 1000원) + (3개 x 1000원) = 5000원
    assert result["2025-01"] == 5000

    # 2025-02 매출 = 1개 x 5000원 = 5000원
    assert result["2025-02"] == 5000


def test_calc_monthly_region_pivot(sample_df):
    df = add_month_column(sample_df)
    pivot = calc_monthly_region_pivot(df)

    # 2025-01, 서울 매출 = (2 x 1000) + (3 x 1000) = 5000원
    assert pivot.loc["2025-01", "서울"] == 5000

    # 2025-02, 부산 매출 = 1 x 5000 = 5000원
    assert pivot.loc["2025-02", "부산"] == 5000

    # 2025-01에는 부산 데이터가 없으므로 결측치(NaN)여야 합니다.
    assert pd.isna(pivot.loc["2025-01", "부산"])
