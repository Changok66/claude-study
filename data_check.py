# data_check.py
# CSV 데이터의 품질(결측치, 이상값, 형식 문제)을 검사하는 함수 모음입니다.

import pandas as pd  # 표 형태(행/열) 데이터를 다루기 위해 pandas를 가져옵니다.


def check_no_missing(df):
    """결측치(빈 값)가 하나라도 있으면 False, 없으면 True를 반환합니다."""
    # df.isnull() : 각 칸이 결측치인지 아닌지를 True/False로 표시한 표를 만듭니다.
    # .sum() : 컬럼별로 True(결측치)의 개수를 더해줍니다.
    missing_counts = df.isnull().sum()

    # missing_counts 중에서 0보다 큰(=결측치가 있는) 컬럼만 골라냅니다.
    missing_columns = missing_counts[missing_counts > 0]

    # 결측치가 있는 컬럼이 하나도 없다면 문제 없음 -> True 반환
    if len(missing_columns) == 0:
        return True

    # 결측치가 있다면 어떤 컬럼에 몇 개 있는지 출력합니다.
    print("[check_no_missing] 결측치가 발견되었습니다:")
    for column_name, count in missing_columns.items():
        print(f" - '{column_name}' 컬럼에 결측치 {count}개")

    return False


def check_no_negative_quantity(df):
    """'수량' 컬럼에 음수 값이 있으면 False, 없으면 True를 반환합니다."""
    # '수량' 컬럼이 아예 없는 경우를 대비해 먼저 확인합니다.
    if "수량" not in df.columns:
        print("[check_no_negative_quantity] '수량' 컬럼이 존재하지 않습니다.")
        return False

    # df["수량"] < 0 : 수량 값이 음수인 행은 True, 아니면 False로 표시됩니다.
    # 이 True/False 표를 이용해 음수인 행만 골라냅니다.
    negative_rows = df[df["수량"] < 0]

    # 음수인 행이 하나도 없다면 문제 없음 -> True 반환
    if len(negative_rows) == 0:
        return True

    # 음수인 행이 있다면 어떤 행(인덱스)에 어떤 값인지 출력합니다.
    print("[check_no_negative_quantity] 음수 수량이 발견되었습니다:")
    for row_index, row in negative_rows.iterrows():
        print(f" - {row_index}번 행: 수량 = {row['수량']}")

    return False


def check_region_names(df):
    """'지역' 컬럼 값에 앞뒤 공백이 있으면 False, 없으면 True를 반환합니다."""
    # '지역' 컬럼이 아예 없는 경우를 대비해 먼저 확인합니다.
    if "지역" not in df.columns:
        print("[check_region_names] '지역' 컬럼이 존재하지 않습니다.")
        return False

    # .str.strip() : 문자열 앞뒤 공백을 제거한 값을 만듭니다.
    # 원래 값과 strip한 값이 다르다면, 앞뒤 공백이 있었다는 뜻입니다.
    #
    # 지역 값이 결측치(NaN)인 행은 여기서 검사하지 않습니다. NaN은 문자열이
    # 아니라서 "공백이 있다/없다"를 따질 수 없고, 결측치 문제는 이미
    # check_no_missing()에서 별도로 확인하기 때문입니다. (그냥 두면 NaN도
    # "공백이 있다"고 잘못 표시됩니다.)
    is_not_missing = df["지역"].notna()
    has_whitespace = is_not_missing & (df["지역"] != df["지역"].str.strip())

    # 공백이 있는 행만 골라냅니다.
    whitespace_rows = df[has_whitespace]

    # 공백이 있는 행이 하나도 없다면 문제 없음 -> True 반환
    if len(whitespace_rows) == 0:
        return True

    # 공백이 있는 행이 있다면 어떤 값인지 출력합니다(공백을 보기 쉽게 대괄호로 표시).
    print("[check_region_names] 앞뒤 공백이 있는 지역 값이 발견되었습니다:")
    for row_index, row in whitespace_rows.iterrows():
        print(f" - {row_index}번 행: '지역' = [{row['지역']}]")

    return False


if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때만 아래 코드가 동작합니다.
    # pd.read_csv() : CSV 파일을 읽어서 DataFrame(표) 형태로 만듭니다.
    # encoding="utf-8-sig" : 한글 CSV를 읽을 때 인코딩 깨짐을 방지하기 위해 지정합니다.
    df = pd.read_csv("data/sales_dirty.csv", encoding="utf-8-sig")

    check_no_missing(df)
    check_no_negative_quantity(df)
    check_region_names(df)
