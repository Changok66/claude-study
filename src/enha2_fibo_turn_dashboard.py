# -*- coding: utf-8 -*-
"""
엔하2+피보턴 백테스트 결과(data/enha2_fibo_turn_backtest.csv)를 실제 캔들
차트 위에 매수(B)/매도(S) 마커로 겹쳐 그리는 대시보드용 함수 모음이다.

plotly 렌더링 유틸(CDN 로딩, 모바일 뷰포트, 확대 시 y축 자동맞춤)과 구름/
피보일목 음영을 그리는 함수는 다른 프로젝트(C:\\Users\\HYEONJEONG\\stock-study)의
나스닥 선물 대시보드(plot_nasdaq_futures_interactive.py,
nasdaq_futures_dashboard_common.py)에서 색상/렌더링 버그까지 이미 다 잡아둔
코드를 그대로 가져왔다(두 프로젝트는 별개의 git 저장소/가상환경이라 import는
못 하고, 코드를 그대로 옮겨적었다). 엔상/엔하 밴드를 그리는 부분은
plot_nasdaq_futures_full_indicators.py의 패턴을 참고하되, 이번엔 밴드
2/3/4/5를 (원본처럼 2·3을 평균내 합치지 않고) 각각 따로 보여달라는 요청이라
그 부분만 새로 맞췄다.

하단(현황카드+거래로그)은 원본과 다르게 새로 짰다 - 원본은 실제 라이브
체결 엑셀에서 "체결 -> 라운드"를 재구성하는 로직(nasdaq_futures_fill_tool.py)에
강하게 묶여 있는데, 우리 데이터는 이미 거래 1건=1행으로 정리된 백테스트
결과라 그런 재구성이 필요 없다. 대신 배색(이익=파랑/손실=회색 등)과 카드/표
스타일은 그대로 베껴왔다.
"""

import math

import plotly.graph_objects as go

from src.ma50_backtest import evaluate_trades


def _profit_factor_text(m):
    """evaluate_trades()가 반환한 손익비가 NaN(손실 거래가 하나도 없거나
    표본이 없을 때)이면 "-"로, 아니면 소수점 2자리 문자열로 바꾼다."""
    pf = m["손익비"]
    return "-" if math.isnan(pf) else f"{pf:.2f}"

# ---------------------------------------------------------------------------
# plotly 렌더링 유틸 (stock-study/nasdaq_futures_dashboard_common.py에서 그대로 이식)
# ---------------------------------------------------------------------------
PLOTLYJS_VERSION = "3.7.0"
PLOTLYJS_CDN = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{PLOTLYJS_VERSION}/plotly.min.js"

VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'

# 휠로 확대하면 화면에 보이는 캔들의 고가/저가에 맞춰 y축(가격)을 다시
# 맞춰주는 스크립트 - 원본에서 "y축을 아예 고정했더니 확대해도 캔들이
# 납작하게 눌려 보이는 문제"를 고치려고 만든 것과 완전히 동일한 코드다.
Y_AUTOFIT_POST_SCRIPT = """
(function() {
    var gd = document.getElementById('{plot_id}');
    if (!gd) return;
    var updating = false;
    function autofitY() {
        if (updating) return;
        var idx = -1;
        for (var i = 0; i < gd.data.length; i++) {
            if (gd.data[i].type === 'candlestick') { idx = i; break; }
        }
        if (idx === -1) return;
        var trace = gd.data[idx];
        var xr = gd.layout.xaxis.range;
        if (!xr) return;
        var x0 = new Date(xr[0]).getTime();
        var x1 = new Date(xr[1]).getTime();
        var lo = Infinity, hi = -Infinity;
        for (var i = 0; i < trace.x.length; i++) {
            var t = new Date(trace.x[i]).getTime();
            if (t >= x0 && t <= x1) {
                if (trace.low[i] < lo) lo = trace.low[i];
                if (trace.high[i] > hi) hi = trace.high[i];
            }
        }
        if (lo === Infinity) return;
        var pad = (hi - lo) * 0.08 || 1;
        updating = true;
        Plotly.relayout(gd, {'yaxis.range': [lo - pad, hi + pad]}).then(function() {
            updating = false;
        });
    }
    gd.on('plotly_relayout', autofitY);
})();
"""


def inject_viewport_meta(html):
    """<head> 바로 뒤에 모바일 뷰포트 meta 태그를 끼워 넣는다."""
    return html.replace("<head>", f"<head>{VIEWPORT_META}", 1)


# ---------------------------------------------------------------------------
# 구름/피보일목 음영 (plot_nasdaq_futures_interactive.py에서 그대로 이식 -
# 2026-09-17에 "일치/혼조 조합" 시도를 되돌리고 확정한, 이미 검증된 버전)
# ---------------------------------------------------------------------------
EN_SANG_COLOR = "235,104,52"   # 엔상(위쪽 밴드) - 주황 계열
EN_HA_COLOR = "74,58,167"      # 엔하(아래쪽 밴드) - 보라 계열
FIBO_POS_FILL = "rgba(208,59,59,0.12)"    # 피보4 > 피보5 (골든)
FIBO_NEG_FILL = "rgba(31,119,212,0.12)"   # 피보4 < 피보5 (데드)


def _add_conditional_region(fig, x, base_values, top_values, cond, color_map, legend_prefix):
    """두 선(base_values/top_values) 사이 영역을, cond 값이 바뀌는 지점마다
    색을 다르게 채운다. color_map={cond값: (fillcolor, 범례라벨)}.
    (stock-study에서 트레이스 폭증 성능 문제까지 고친 버전을 그대로 가져옴 -
    상태 전환마다 트레이스를 새로 만들지 않고, 같은 상태의 구간들을 트레이스
    하나에 몰아넣는다)
    """
    valid = base_values.notna() & top_values.notna() & cond.notna()
    if not valid.all():
        x = x[valid]
        base_values = base_values[valid]
        top_values = top_values[valid]
        cond = cond[valid]
    n = len(x)
    if n == 0:
        return

    cond_arr = cond.to_numpy()
    boundaries = [0]
    for i in range(1, n):
        if cond_arr[i] != cond_arr[i - 1]:
            boundaries.append(i)
    boundaries.append(n)

    groups = {key: {"x": [], "base": [], "top": []} for key in color_map}
    for k in range(len(boundaries) - 1):
        s = boundaries[k]
        e = min(boundaries[k + 1] + 1, n)
        key = cond_arr[s]
        g = groups[key]
        if g["x"]:
            g["x"].append(None)
            g["base"].append(None)
            g["top"].append(None)
        g["x"].extend(x[s:e])
        g["base"].extend(base_values.iloc[s:e])
        g["top"].extend(top_values.iloc[s:e])

    for key, (fillcolor, label) in color_map.items():
        g = groups[key]
        if not g["x"]:
            continue
        fig.add_trace(go.Scatter(
            x=g["x"], y=g["base"], mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
            connectgaps=False,
        ))
        fig.add_trace(go.Scatter(
            x=g["x"], y=g["top"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor=fillcolor,
            name=label, showlegend=True, hoverinfo="skip",
            connectgaps=False,
        ))


def _add_envelope_bands(fig, df_plot):
    """
    엔상1(중심)~5, 엔하2~5를 구름으로 그린다. 원본(나스닥 대시보드)은 2/3을
    평균내 하나로 합쳐서 3겹으로 줄였지만, 이번엔 "엔상/엔하 밴드(2,3,4,5)
    표시"를 요청받았으므로 4개 밴드(1->2, 2->3, 3->4, 4->5)를 각각 따로
    구분해서 그린다.
    """
    x = df_plot.index
    upper_lines = ["엔상1", "엔상2", "엔상3", "엔상4", "엔상5"]
    lower_lines = ["엔상1", "엔하2", "엔하3", "엔하4", "엔하5"]
    band_alphas = [0.22, 0.16, 0.10, 0.05]  # 중심에서 멀어질수록 옅어짐

    for i in range(4):
        base_col, top_col = upper_lines[i], upper_lines[i + 1]
        fig.add_trace(go.Scatter(
            x=x, y=df_plot[base_col], mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=df_plot[top_col], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=f"rgba({EN_SANG_COLOR},{band_alphas[i]})",
            name="엔상 구름(1~5)", legendgroup="ensang", showlegend=(i == 0),
            hoverinfo="skip",
        ))

    for i in range(4):
        base_col, bottom_col = lower_lines[i], lower_lines[i + 1]
        fig.add_trace(go.Scatter(
            x=x, y=df_plot[base_col], mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=df_plot[bottom_col], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=f"rgba({EN_HA_COLOR},{band_alphas[i]})",
            name="엔하 구름(1~5)", legendgroup="enha", showlegend=(i == 0),
            hoverinfo="skip",
        ))

    # 실제 값 선 - 엔상1(중심선)만 진하게, 2~5는 각 밴드를 구분할 수 있도록
    # 옅게 따로 그린다(요청한 "밴드 2,3,4,5 표시"를 hover로 정확히 값까지
    # 확인할 수 있게 하기 위함). 이번 신호의 핵심인 엔하2만 다른 밴드보다
    # 굵게 강조한다.
    fig.add_trace(go.Scatter(
        x=x, y=df_plot["엔상1"], name="엔상1(MA50)",
        line=dict(color="#333333", width=1.3),
        hovertemplate="엔상1 %{y:.2f}<extra></extra>",
    ))
    for col in ["엔상2", "엔상3", "엔상4", "엔상5"]:
        fig.add_trace(go.Scatter(
            x=x, y=df_plot[col], name=col, opacity=0.6,
            line=dict(color=f"rgb({EN_SANG_COLOR})", width=1, dash="dot"),
            legendgroup="ensang_line", showlegend=False,
            hovertemplate=f"{col} " + "%{y:.2f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=df_plot["엔하2"], name="엔하2 (이번 신호 진입 밴드)",
        line=dict(color=f"rgb({EN_HA_COLOR})", width=2),
        hovertemplate="엔하2 %{y:.2f}<extra></extra>",
    ))
    for col in ["엔하3", "엔하4", "엔하5"]:
        fig.add_trace(go.Scatter(
            x=x, y=df_plot[col], name=col, opacity=0.6,
            line=dict(color=f"rgb({EN_HA_COLOR})", width=1, dash="dot"),
            legendgroup="enha_line", showlegend=False,
            hovertemplate=f"{col} " + "%{y:.2f}<extra></extra>",
        ))


def _add_fibo_lines(fig, df_plot):
    """피보4/피보5 선 + 골든/데드 상태에 따른 음영 (원본과 동일한 배색)."""
    x = df_plot.index
    _add_conditional_region(
        fig, x, df_plot["피보5"], df_plot["피보4"],
        df_plot["피보4"] >= df_plot["피보5"],
        {True: (FIBO_POS_FILL, "피보일목(골든)"), False: (FIBO_NEG_FILL, "피보일목(데드)")},
        "피보일목",
    )
    fig.add_trace(go.Scatter(
        x=x, y=df_plot["피보4"], name="피보4",
        line=dict(color="#eb6834", width=1, dash="dash"),
        hovertemplate="피보4 %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_plot["피보5"], name="피보5",
        line=dict(color="#4a3aa7", width=1, dash="dash"),
        hovertemplate="피보5 %{y:.2f}<extra></extra>",
    ))


def _compute_exit_dates(df, trades):
    """
    거래 표(trades)의 각 행에 대해 "진입일(터치시각) + 보유봉수"번째 봉의
    날짜(=청산일)를 계산해서 리스트로 반환한다. CSV에는 청산일이 직접
    저장돼 있지 않아서(진입가/청산가/보유봉수만 있음) 여기서 다시 찾는다.

    타임존 표기 차이(문자열로 저장됐다 다시 읽으면서 생길 수 있는 미세한
    차이) 문제를 피하려고, Timestamp를 그대로 비교하지 않고 "날짜(date)"
    기준으로 df 안에서의 위치를 찾는다.
    """
    date_to_pos = {ts.date(): i for i, ts in enumerate(df.index)}
    exit_dates = []
    for _, row in trades.iterrows():
        entry_date = row["터치시각"].date()
        pos = date_to_pos[entry_date]
        exit_pos = min(pos + int(row["보유봉수"]), len(df) - 1)
        exit_dates.append(df.index[exit_pos])
    return exit_dates


def _add_bs_markers(fig, df, trades):
    """매수(B)/매도(S) 마커를 추가한다. 매도는 손익률(%) 부호로 익절(금색)/
    손절(회색)을 나눈다 (plot_nasdaq_futures_interactive.py의 배색과 동일)."""
    trades = trades.copy()
    trades["청산일"] = _compute_exit_dates(df, trades)

    fig.add_trace(go.Scatter(
        x=trades["터치시각"], y=trades["진입가"], name="매수(B)",
        mode="markers+text", text=["B"] * len(trades), textposition="middle center",
        textfont=dict(color="white", size=10, family="Malgun Gothic"),
        marker=dict(symbol="circle", size=18, color="#0ca30c", line=dict(color="white", width=2)),
        hovertemplate="매수 %{x|%Y-%m-%d} @ %{y:.2f}<extra></extra>",
    ))

    def _add_sell_trace(mask, name, color):
        g = trades[mask]
        if g.empty:
            return
        fig.add_trace(go.Scatter(
            x=g["청산일"], y=g["청산가"], name=name,
            mode="markers+text", text=["S"] * len(g), textposition="middle center",
            textfont=dict(color="#1a1a1a", size=10, family="Malgun Gothic"),
            marker=dict(symbol="circle", size=18, color=color, line=dict(color="white", width=2)),
            customdata=list(zip(g["손익률(%)"], g["청산사유"])),
            hovertemplate=(
                f"{name} %{{x|%Y-%m-%d}} @ %{{y:.2f}}"
                "<br>손익률 %{customdata[0]:+.2f}% (%{customdata[1]})"
                "<extra></extra>"
            ),
        ))

    _add_sell_trace(trades["손익률(%)"] > 0, "매도(익절)", "#e0a600")
    _add_sell_trace(trades["손익률(%)"] <= 0, "매도(손절)", "#b0b0b0")


def build_price_figure(df, trades, label):
    """캔들 + 엔상/엔하 밴드(2~5) + 피보일목(피보4/5) + 매수/매도 마커를
    그린 plotly Figure를 반환한다."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=label, increasing_line_color="#d03b3b", decreasing_line_color="#1f77d4",
    ))

    _add_envelope_bands(fig, df)
    _add_fibo_lines(fig, df)
    _add_bs_markers(fig, df, trades)

    fig.update_traces(selector=dict(type="candlestick"), hoverlabel=dict(font=dict(family="Malgun Gothic")))
    fig.update_layout(
        title=f"{label} 일봉 - 엔하2+피보턴 신호 {len(trades)}건 "
              "(buy=B 초록, sell=S 금색(익절)/회색(손절)) - 휠로 확대/축소, 드래그로 이동",
        xaxis_title="날짜", yaxis_title="가격",
        xaxis_range=[df.index.min(), df.index.max()],
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        font=dict(family="Malgun Gothic"),
        width=None, height=700, autosize=True,
        dragmode="pan", hovermode="closest",
    )
    return fig


def embed_figure_html(fig):
    """Y축 자동맞춤 스크립트 + CDN 방식으로 plotly.js를 불러오는 HTML 조각을 반환한다."""
    return fig.to_html(
        full_html=False, include_plotlyjs=PLOTLYJS_CDN,
        config={"scrollZoom": True, "responsive": True},
        default_width="100%",
        post_script=Y_AUTOFIT_POST_SCRIPT,
    )


# ---------------------------------------------------------------------------
# 하단: 현황카드 + 거래로그 (새로 작성, 배색만 stock-study 관례를 그대로 따름)
# ---------------------------------------------------------------------------
CARD_BG = "#F2F2F2"
CARD_LABEL_COLOR = "#595959"
CARD_VALUE_COLOR = "#1F4E78"
PROFIT_COLOR = "#0070C0"
LOSS_COLOR = "#808080"

DASHBOARD_STYLE = f"""
<style>
  body {{ margin: 0; }}
  .dashboard-bottom {{
    font-family: "Malgun Gothic", sans-serif;
    max-width: 1400px;
    margin: 12px auto 40px;
    padding: 0 8px;
  }}
  .status-card {{
    display: flex; flex-wrap: wrap; gap: 0;
    background: {CARD_BG}; border-radius: 6px;
    padding: 14px 20px; margin-bottom: 16px;
  }}
  .card-item {{ flex: 1 1 160px; padding: 6px 14px; }}
  .card-item.wide {{ flex: 1 1 320px; }}
  .card-label {{ color: {CARD_LABEL_COLOR}; font-weight: bold; font-size: 13px; }}
  .card-value {{ color: {CARD_VALUE_COLOR}; font-weight: bold; font-size: 16px; margin-top: 2px; }}
  table.tradelog {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.tradelog thead th {{
    background: #D9E1F2; font-weight: bold; padding: 6px 8px;
    position: sticky; top: 0;
  }}
  table.tradelog td {{ padding: 5px 8px; border-bottom: 1px solid #eee; }}
  table.tradelog td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .tradelog-scroll {{
    max-height: 500px; overflow-y: auto;
    border: 1px solid #ddd; border-radius: 4px;
  }}
  h2.section-title {{ font-family: "Malgun Gothic", sans-serif; margin: 18px 0 8px; }}
</style>
"""


def _status_card_html(trades, label):
    m_all = evaluate_trades(trades)
    profit_factor_text = _profit_factor_text(m_all)
    avg_hold = trades["보유봉수"].mean()

    rows = ""
    for 구간 in ["학습", "검증"]:
        g = trades[trades["구간"] == 구간]
        if g.empty:
            continue
        m = evaluate_trades(g)
        pf_text = _profit_factor_text(m)
        rows += (
            f'<div class="card-item"><div class="card-label">{구간} 승률/손익비</div>'
            f'<div class="card-value">{m["승률(%)"]:.1f}% / {pf_text} ({m["거래수"]}건)</div></div>'
        )

    return f"""
<div class="status-card">
  <div class="card-item wide"><div class="card-label">종목</div><div class="card-value">{label}</div></div>
  <div class="card-item"><div class="card-label">전체 거래건수</div><div class="card-value">{m_all["거래수"]}건</div></div>
  <div class="card-item"><div class="card-label">전체 승률</div><div class="card-value">{m_all["승률(%)"]:.1f}%</div></div>
  <div class="card-item"><div class="card-label">전체 손익비</div><div class="card-value">{profit_factor_text}</div></div>
  <div class="card-item"><div class="card-label">평균 보유봉수</div><div class="card-value">{avg_hold:.1f}봉</div></div>
  {rows}
</div>
"""


def _tradelog_table_html(trades):
    ordered = trades.sort_values("터치시각", ascending=False)
    rows_html = []
    for _, r in ordered.iterrows():
        pnl = r["손익률(%)"]
        color = PROFIT_COLOR if pnl > 0 else LOSS_COLOR
        rows_html.append(
            "<tr>"
            f'<td>{r["터치시각"]:%Y-%m-%d}</td>'
            f'<td class="num">{r["진입가"]:.2f}</td>'
            f'<td class="num">{r["청산가"]:.2f}</td>'
            f'<td>{r["청산사유"]}</td>'
            f'<td class="num">{int(r["보유봉수"])}</td>'
            f'<td class="num" style="color:{color}">{pnl:+.2f}%</td>'
            f'<td>{r["구간"]}</td>'
            "</tr>"
        )

    headers = ["진입일", "진입가", "청산가", "청산사유", "보유봉수", "손익률(%)", "구간"]
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    return f"""
<table class="tradelog">
  <thead><tr>{header_html}</tr></thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>
"""


def render_bottom_section_html(trades, label):
    """상단 차트 아래에 붙일 현황카드+거래로그 섹션 전체(스타일 포함)를 반환한다."""
    card_html = _status_card_html(trades, label)
    table_html = _tradelog_table_html(trades)
    return f"""
{DASHBOARD_STYLE}
<div class="dashboard-bottom">
  {card_html}
  <h2 class="section-title">거래로그 (전체 {len(trades)}건)</h2>
  <div class="tradelog-scroll">{table_html}</div>
</div>
"""
