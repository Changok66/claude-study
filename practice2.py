# practice2.py
# 지저분한(정리되지 않은) 숫자 데이터를 정리하는 연습 코드입니다.
# 처리하는 문제: 1) 빈 값 제거  2) 이상치 제거  3) 중복 제거

# 1) 일부러 지저분하게 만든 원본 데이터입니다.
#    - None: 값이 없는 빈 데이터 (예: 설문에 응답하지 않은 항목)
#    - 1000: 다른 값들에 비해 지나치게 큰 이상치
#    - -5: 정상 범위를 벗어난 이상치(음수)
#    - 12, 45가 두 번씩 등장해서 중복이 있는 상태입니다.
raw_data = [23, 45, None, 12, 67, 12, None, 34, 89, 1000, 21, 45, -5]

print(f"원본 데이터: {raw_data}")

# 2) 빈 값(None)을 제거합니다.
#    - 리스트 컴프리헨션([x for x in ... if 조건]): 조건을 만족하는 값만 골라 새 리스트를 만드는 문법
#    - "x is not None": x가 None이 아닌 경우만 남기라는 뜻
no_empty = [x for x in raw_data if x is not None]
print(f"빈 값 제거 후: {no_empty}")

# 3) 이상치를 제거합니다.
#    - 이상치란 다른 값들에 비해 지나치게 크거나 작아서 비정상적으로 보이는 값입니다.
#    - 여기서는 이해하기 쉽도록 "정상 범위"를 0 이상 200 이하로 직접 정해두고,
#      그 범위를 벗어나는 값을 이상치로 간주해 제거합니다.
LOWER_BOUND = 0
UPPER_BOUND = 200
no_outliers = [x for x in no_empty if LOWER_BOUND <= x <= UPPER_BOUND]
print(f"이상치 제거 후: {no_outliers}")

# 4) 중복된 값을 제거합니다.
#    - seen: 지금까지 등장한 값을 기억해두는 set(집합) 자료구조
#      set은 같은 값을 중복으로 저장하지 않는 특징이 있어서 "이미 나온 값인지" 빠르게 확인할 수 있습니다.
#    - 순서를 유지하면서 중복만 제거하기 위해 반복문(for)으로 하나씩 확인합니다.
seen = set()
cleaned_data = []
for value in no_outliers:
    if value not in seen:      # 아직 등장하지 않은 값이라면
        seen.add(value)        # seen에 기록해두고
        cleaned_data.append(value)  # 결과 리스트에 추가합니다.

# 5) 최종 정리된 데이터를 출력합니다.
print(f"최종 정리된 데이터: {cleaned_data}")
