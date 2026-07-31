# -*- coding: utf-8 -*-
"""
data/sales.csv를 읽어 지역별 총 매출(수량 x 단가)을 계산하고
막대그래프로 그려 images 폴더에 저장한다.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# 그래프에서 한글이 깨지지 않도록 폰트 설정 (Windows 기본 한글 폰트)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = os.path.join("data", "sales.csv")
IMAGE_DIR = "images"
IMAGE_PATH = os.path.join(IMAGE_DIR, "region_sales.png")


def load_sales(path):
    # CSV를 읽어 데이터프레임으로 반환
    return pd.read_csv(path, encoding="utf-8-sig")


def calc_region_totals(df):
    # 행별 매출(수량 x 단가)을 구한 뒤 지역별로 합산
    df = df.copy()
    df["매출"] = df["수량"] * df["단가"]
    return df.groupby("지역")["매출"].sum().sort_values(ascending=False)


def plot_region_totals(region_totals, output_path):
    # 지역별 총 매출을 막대그래프로 그려서 파일로 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(region_totals.index, region_totals.values, color="steelblue")
    ax.set_title("지역별 총 매출")
    ax.set_xlabel("지역")
    ax.set_ylabel("총 매출(원)")
    ax.ticklabel_format(style="plain", axis="y")  # y축 숫자를 지수 표기 없이 표시

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    df = load_sales(DATA_PATH)
    region_totals = calc_region_totals(df)
    plot_region_totals(region_totals, IMAGE_PATH)
    print(f"{IMAGE_PATH} 저장 완료")
    print(region_totals)


if __name__ == "__main__":
    main()
