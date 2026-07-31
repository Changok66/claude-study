# -*- coding: utf-8 -*-
"""
파이썬 초보자를 위한 기초 예제
변수, 숫자/문자열/불리언, 리스트, 딕셔너리
"""

import sys

# Windows 콘솔(cmd, PowerShell 등)에서 한글 출력이 깨지는 것을 방지
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


# ==========================================
# 1. 변수 (Variable)
# ==========================================
print("=" * 40)
print("1. 변수")
print("=" * 40)

# 변수는 값을 저장하는 상자와 같음
name = "홍길동"   # 변수에 문자열 값을 저장
age = 20          # 변수에 숫자 값을 저장

print("이름:", name)
print("나이:", age)


# ==========================================
# 2. 숫자 타입 (int, float)
# ==========================================
print()
print("=" * 40)
print("2. 숫자 타입")
print("=" * 40)

integer_num = 10       # 정수(int)
float_num = 3.14       # 실수(float)

print("정수:", integer_num, "| 타입:", type(integer_num))
print("실수:", float_num, "| 타입:", type(float_num))
print("덧셈 결과:", integer_num + float_num)


# ==========================================
# 3. 문자열 타입 (str)
# ==========================================
print()
print("=" * 40)
print("3. 문자열 타입")
print("=" * 40)

greeting = "안녕하세요"        # 문자열은 큰따옴표(") 또는 작은따옴표(')로 표현
subject = '파이썬'

# + 연산자로 문자열을 이어 붙일 수 있음
message = greeting + ", " + subject + "!"
print(message)
print("문자열 길이:", len(message))


# ==========================================
# 4. 불리언 타입 (bool)
# ==========================================
print()
print("=" * 40)
print("4. 불리언 타입")
print("=" * 40)

is_student = True      # 참(True) 또는 거짓(False)만 가질 수 있음
is_adult = age >= 20    # 비교 연산 결과도 불리언 값이 됨

print("학생 여부:", is_student)
print("성인 여부:", is_adult)


# ==========================================
# 5. 리스트 (list)
# ==========================================
print()
print("=" * 40)
print("5. 리스트")
print("=" * 40)

# 리스트는 여러 값을 순서대로 담는 자료형 (대괄호 [] 사용)
fruits = ["사과", "바나나", "딸기"]

print("과일 목록:", fruits)
print("첫 번째 과일:", fruits[0])   # 인덱스는 0부터 시작

fruits.append("포도")               # 리스트 맨 뒤에 값 추가
print("포도 추가 후:", fruits)
print("리스트 길이:", len(fruits))


# ==========================================
# 6. 딕셔너리 (dict)
# ==========================================
print()
print("=" * 40)
print("6. 딕셔너리")
print("=" * 40)

# 딕셔너리는 키(key)와 값(value)의 쌍으로 이루어진 자료형 (중괄호 {} 사용)
person = {
    "이름": "홍길동",
    "나이": 20,
    "직업": "학생",
}

print("사람 정보:", person)
print("이름 조회:", person["이름"])

person["나이"] = 21   # 키를 이용해 값 수정
print("나이 수정 후:", person)

# 모든 키와 값을 순회하며 출력
for key, value in person.items():
    print(f"{key}: {value}")
