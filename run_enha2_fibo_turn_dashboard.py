# -*- coding: utf-8 -*-
"""
엔하2+피보턴 백테스트 결과(data/enha2_fibo_turn_backtest.csv, 608건)를 실제
캔들 차트 위에 매수(B)/매도(S) 마커로 겹쳐서 보여주는 대시보드를 9종목 각각
개별 HTML로 만드는 실행 스크립트다.

차트 렌더링(캔들+엔상/엔하 밴드+피보일목+마커, plotly CDN 로딩, 확대 시 y축
자동맞춤)과 하단(현황카드+거래로그) 조립은 전부 src/enha2_fibo_turn_dashboard.py에
있다 - 이 파일은 종목을 순회하며 데이터를 준비하고 HTML로 저장하는 역할만 한다.

생성되는 결과 파일: dashboards/엔하2_피보턴_{종목명}.html (9개)
"""

from pathlib import Path

import pandas as pd

from src.enha2_fibo_turn_dashboard import (
    build_price_figure,
    embed_figure_html,
    inject_viewport_meta,
    render_bottom_section_html,
)
from src.ma50_data import fetch_kr_daily_bars
from src.reversal_angle import add_all_indicators

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_PATH = BASE_DIR / "data" / "enha2_fibo_turn_backtest.csv"
OUTPUT_DIR = BASE_DIR / "dashboards"


def build_dashboard_html(df, trades, label):
    fig = build_price_figure(df, trades, label)
    chart_html = embed_figure_html(fig)
    bottom_html = render_bottom_section_html(trades, label)

    title = f"{label} - 엔하2+피보턴 백테스트 대시보드"
    html = (
        '<html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        f"</head><body>"
        f'<div style="font-family:\'Malgun Gothic\',sans-serif; padding:10px 16px 0;">'
        f'<h1 style="margin:0; font-size:20px;">{title}</h1></div>'
        f"{chart_html}"
        f"{bottom_html}"
        f"</body></html>"
    )
    return inject_viewport_meta(html)


def main():
    print(f"[1] 백테스트 결과 로딩: {BACKTEST_PATH}")
    all_trades = pd.read_csv(BACKTEST_PATH, encoding="utf-8-sig")
    # CSV에는 시각이 문자열로 저장돼 있으므로 다시 datetime으로 바꾼다.
    all_trades["터치시각"] = pd.to_datetime(all_trades["터치시각"])

    OUTPUT_DIR.mkdir(exist_ok=True)

    for label, trades in all_trades.groupby("종목명"):
        print(f"[INFO] {label} 대시보드 생성 중... ({len(trades)}건)")
        ticker, raw = fetch_kr_daily_bars(label)
        if raw.empty:
            print(f"  [경고] {label} 데이터를 받아오지 못했습니다. 건너뜁니다.")
            continue

        df = add_all_indicators(raw)
        html = build_dashboard_html(df, trades, label)

        output_path = OUTPUT_DIR / f"엔하2_피보턴_{label}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  -> 저장 완료: {output_path}")

    print(f"\n[완료] {OUTPUT_DIR} 폴더에 9개 대시보드 HTML을 생성했습니다.")


if __name__ == "__main__":
    main()
