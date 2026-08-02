# test_data_check.py
# data_check.py에 있는 세 가지 검사 함수를 pytest로 검증하는 테스트 파일입니다.

from pathlib import Path  # 이 테스트 파일 기준으로 데이터 폴더 경로를 계산하기 위해 사용합니다.

import pandas as pd  # CSV를 읽어 DataFrame으로 만들기 위해 pandas를 가져옵니다.
import pytest  # 테스트 함수를 작성하고 실행하기 위해 pytest를 가져옵니다.

# data_check.py에서 검사할 세 함수를 가져옵니다.
from data_check import (
    check_no_missing,
    check_no_negative_quantity,
    check_region_names,
)

# "data/sales.csv"처럼 이름만 적으면, pytest를 프로젝트 루트가 아닌 다른 폴더에서
# 실행했을 때 파일을 찾지 못해 테스트가 실패한다. 이 테스트 파일(__file__)의 위치를
# 기준으로 절대경로를 만들어서, 어디서 pytest를 실행하든 항상 같은 데이터를 읽도록 한다.
DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture
def clean_df():
    """깨끗한 데이터(sales.csv)를 읽어서 DataFrame으로 반환하는 fixture입니다."""
    # fixture로 만들어두면 여러 테스트 함수에서 매번 다시 읽지 않고 재사용할 수 있습니다.
    return pd.read_csv(DATA_DIR / "sales.csv", encoding="utf-8-sig")


@pytest.fixture
def dirty_df():
    """더러운 데이터(sales_dirty.csv)를 읽어서 DataFrame으로 반환하는 fixture입니다."""
    return pd.read_csv(DATA_DIR / "sales_dirty.csv", encoding="utf-8-sig")


# ---- sales.csv(깨끗한 데이터)는 세 검사 모두 통과(True)해야 합니다 ----


def test_clean_data_has_no_missing(clean_df):
    # 깨끗한 데이터에는 결측치가 없어야 하므로 True가 나와야 합니다.
    assert check_no_missing(clean_df) is True


def test_clean_data_has_no_negative_quantity(clean_df):
    # 깨끗한 데이터에는 음수 수량이 없어야 하므로 True가 나와야 합니다.
    assert check_no_negative_quantity(clean_df) is True


def test_clean_data_has_valid_region_names(clean_df):
    # 깨끗한 데이터에는 지역 값에 공백이 없어야 하므로 True가 나와야 합니다.
    assert check_region_names(clean_df) is True


# ---- sales_dirty.csv(더러운 데이터)는 세 검사 모두 실패(False)해야 합니다 ----


def test_dirty_data_has_missing(dirty_df):
    # 더러운 데이터에는 결측치(수량 빈 값)가 있으므로 False가 나와야 합니다.
    assert check_no_missing(dirty_df) is False


def test_dirty_data_has_negative_quantity(dirty_df):
    # 더러운 데이터에는 음수 수량(-1)이 있으므로 False가 나와야 합니다.
    assert check_no_negative_quantity(dirty_df) is False


def test_dirty_data_has_invalid_region_names(dirty_df):
    # 더러운 데이터에는 앞뒤 공백이 있는 지역 값("서울 ")이 있으므로 False가 나와야 합니다.
    assert check_region_names(dirty_df) is False
