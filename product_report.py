# -*- coding: utf-8 -*-
"""
data/sales.csv를 읽어 '제품별' 총 매출을 계산하고,
매출이 높은 순서대로 막대그래프를 그려 images 폴더에 저장한다.

CSV를 읽는 load_sales 함수는 이미 analyze.py에 만들어져 있으므로
새로 만들지 않고 그대로 가져다 쓴다 (같은 로직을 두 번 만들지 않기 위함).
"""

import os

import matplotlib.pyplot as plt

# analyze.py에 이미 만들어져 있는 함수를 가져와서 재사용한다.
# report.py도 같은 방식으로 이 함수를 가져다 쓰고 있다.
from analyze import load_sales

# 그래프에서 한글이 깨지지 않도록 폰트 설정 (analyze.py, report.py와 동일하게 설정)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = os.path.join("data", "sales.csv")
IMAGE_DIR = "images"
IMAGE_PATH = os.path.join(IMAGE_DIR, "product_sales.png")


def calc_product_totals(df):
    # 원본 데이터프레임을 바꾸지 않기 위해 복사본을 만든다.
    df = df.copy()

    # 행별 매출 = 수량 x 단가
    df["매출"] = df["수량"] * df["단가"]

    # groupby("제품") : 같은 제품끼리 묶는다.
    # ["매출"].sum() : 묶인 그룹마다 매출을 더한다.
    # sort_values(ascending=False) : 매출이 큰 제품부터 내림차순으로 정렬한다.
    return df.groupby("제품")["매출"].sum().sort_values(ascending=False)


def plot_product_totals(product_totals, output_path):
    # images 폴더가 없으면 새로 만든다 (이미 있으면 그냥 넘어간다).
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # product_totals는 이미 매출 내림차순으로 정렬되어 있으므로,
    # index(제품명) 순서 그대로 막대그래프를 그리면 매출 높은 순으로 표시된다.
    ax.bar(product_totals.index, product_totals.values, color="steelblue")
    ax.set_title("제품별 총 매출 (높은 순)")
    ax.set_xlabel("제품")
    ax.set_ylabel("총 매출(원)")
    ax.ticklabel_format(style="plain", axis="y")  # y축 숫자를 지수 표기 없이 표시

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    df = load_sales(DATA_PATH)
    product_totals = calc_product_totals(df)

    plot_product_totals(product_totals, IMAGE_PATH)

    print("=== 제품별 총 매출 (높은 순) ===")
    print(product_totals)
    print()
    print(f"{IMAGE_PATH} 저장 완료")


if __name__ == "__main__":
    main()
