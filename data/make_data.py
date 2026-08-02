import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path

# 이 스크립트는 data/ 폴더 안에 있지만, 지역/제품 이름 목록은 src/metrics.py에서
# "단일 기준"으로 관리한다 (REGION_ORDER, PRODUCT_ORDER). 실행 위치(현재 작업 폴더)와
# 무관하게 src 패키지를 찾을 수 있도록, 프로젝트 루트 폴더를 모듈 검색 경로에 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import PRODUCT_ORDER, REGION_ORDER  # noqa: E402

random.seed(42)

REGIONS = REGION_ORDER

PRODUCTS = {
    "노트북": (800000, 1500000),
    "마우스": (10000, 30000),
    "키보드": (20000, 60000),
    "모니터": (150000, 400000),
}

# PRODUCTS의 키(제품 이름)가 src/metrics.py의 PRODUCT_ORDER와 정확히 일치하는지 확인한다.
# 나중에 제품이 추가/변경됐는데 한쪽을 깜빡 고치지 않으면, 그래프를 그릴 때가 되어서야
# 알아보기 힘든 에러가 나는 대신 데이터 생성 시점에 바로 명확한 메시지로 알려준다.
assert set(PRODUCTS.keys()) == set(PRODUCT_ORDER), (
    "PRODUCTS의 제품 목록이 src/metrics.py의 PRODUCT_ORDER와 다릅니다. "
    "두 목록을 같은 제품 이름으로 맞춰주세요."
)

START_DATE = date(2025, 1, 1)
NUM_ROWS = 200

def random_date():
    offset = random.randint(0, 364)
    return START_DATE + timedelta(days=offset)

def generate_rows(n):
    rows = []
    for _ in range(n):
        product = random.choice(list(PRODUCTS.keys()))
        min_price, max_price = PRODUCTS[product]
        row = {
            "날짜": random_date().isoformat(),
            "지역": random.choice(REGIONS),
            "제품": product,
            "수량": random.randint(1, 20),
            "단가": random.randrange(min_price, max_price, 1000),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["날짜"])
    return rows

def main():
    rows = generate_rows(NUM_ROWS)
    # "sales.csv"처럼 이름만 적으면 스크립트를 실행하는 위치(현재 작업 폴더)에 따라
    # 엉뚱한 곳에 파일이 생길 수 있다. Path(__file__)를 기준으로 항상 이 스크립트와
    # 같은 폴더(data/)에 저장되도록 절대경로를 만든다.
    output_path = Path(__file__).resolve().parent / "sales.csv"
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["날짜", "지역", "제품", "수량", "단가"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output_path} 생성 완료 ({len(rows)}행)")

if __name__ == "__main__":
    main()
