# -*- coding: utf-8 -*-
"""
파이썬 초보자를 위한 기초 예제
if/elif/else 조건문, for 반복문, def 함수 정의
"""

import sys

# Windows 콘솔(cmd, PowerShell 등)에서 한글 출력이 깨지는 것을 방지
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


# ==========================================
# 1. 조건문 (if / elif / else)
# ==========================================
print("=" * 40)
print("1. 조건문")
print("=" * 40)

score = 85

# 조건에 따라 실행할 코드를 분기함
if score >= 90:
    grade = "A"
elif score >= 80:      # 위 조건이 거짓일 때 다음 조건을 검사
    grade = "B"
elif score >= 70:
    grade = "C"
else:                   # 모든 조건이 거짓일 때 실행
    grade = "F"

print(f"점수: {score}, 학점: {grade}")


# ==========================================
# 2. 반복문 (for)
# ==========================================
print()
print("=" * 40)
print("2. 반복문")
print("=" * 40)

# range(n)은 0부터 n-1까지의 숫자를 순서대로 만들어냄
for i in range(5):
    print(f"{i}번째 반복")

print()

# 리스트의 각 요소를 순서대로 꺼내며 반복
fruits = ["사과", "바나나", "딸기"]
for fruit in fruits:
    print("과일:", fruit)


# ==========================================
# 3. 함수 정의 (def)
# ==========================================
print()
print("=" * 40)
print("3. 함수 정의")
print("=" * 40)


def greet(name):
    """이름을 받아 인사말을 반환하는 함수"""
    return f"안녕하세요, {name}님!"


def add(a, b):
    """두 수를 더한 값을 반환하는 함수"""
    return a + b


print(greet("홍길동"))
print("3 + 5 =", add(3, 5))


# ==========================================
# 4. 조합 예제 (조건문 + 반복문 + 함수)
# ==========================================
print()
print("=" * 40)
print("4. 조합 예제")
print("=" * 40)


def get_grade(score):
    """점수를 받아 학점을 반환하는 함수"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"


scores = [95, 82, 74, 60, 88]

# 점수 목록을 순회하면서 각 점수의 학점을 함수로 계산해 출력
for score in scores:
    grade = get_grade(score)
    print(f"점수 {score} -> 학점 {grade}")
