import csv
import random
from datetime import date, timedelta

random.seed(42)

REGIONS = ["서울", "부산", "대구", "인천"]

PRODUCTS = {
    "노트북": (800000, 1500000),
    "마우스": (10000, 30000),
    "키보드": (20000, 60000),
    "모니터": (150000, 400000),
}

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
    output_path = "sales.csv"
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["날짜", "지역", "제품", "수량", "단가"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output_path} 생성 완료 ({len(rows)}행)")

if __name__ == "__main__":
    main()
