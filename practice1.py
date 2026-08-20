# practice1.py
# 숫자 리스트의 평균과 최댓값을 구해서 출력하는 연습 코드입니다.

# statistics 모듈: 중앙값(median) 등 통계 계산을 도와주는 파이썬 표준 라이브러리입니다.
# (별도 설치 없이 바로 사용할 수 있습니다.)
import statistics

# 1) 숫자들을 담은 리스트를 만듭니다.
numbers = [23, 45, 12, 67, 34, 89, 21]

# 2) 합계를 구합니다.
#    - sum(numbers): 리스트 안의 모든 숫자를 더한 값
total = sum(numbers)

# 2-1) 평균을 구합니다.
#    - len(numbers): 리스트 안에 들어있는 숫자의 개수
#    - 평균 = 전체 합 / 개수
average = total / len(numbers)

# 3) 최댓값을 구합니다.
#    - max(numbers): 리스트 안에서 가장 큰 값을 찾아주는 파이썬 내장 함수
maximum = max(numbers)

# 3-1) 최솟값을 구합니다.
#    - min(numbers): 리스트 안에서 가장 작은 값을 찾아주는 파이썬 내장 함수
minimum = min(numbers)

# 4) 평균값을 소수점 둘째 자리까지 반올림합니다.
#    - round(값, 자릿수): 값을 지정한 소수점 자릿수까지 반올림해주는 파이썬 내장 함수
average_rounded = round(average, 2)

# 4-1) 중앙값(median)을 구합니다.
#    - statistics.median(numbers): 리스트를 정렬했을 때 한가운데에 오는 값을 구해주는 함수
#    - 숫자 개수가 짝수면 가운데 두 값의 평균을 자동으로 계산해줍니다.
median_value = statistics.median(numbers)

# 5) 결과를 화면에 출력합니다.
#    - f"..." 형태는 f-string이라고 부르며, 문자열 안에 변수 값을 쉽게 넣을 수 있게 해줍니다.
print(f"숫자 리스트: {numbers}")
print(f"합계: {total}")
print(f"평균: {average_rounded}")
print(f"최댓값: {maximum}")
print(f"최솟값: {minimum}")
print(f"중앙값: {median_value}")
