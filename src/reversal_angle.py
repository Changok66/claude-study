# -*- coding: utf-8 -*-
"""
엔상5/엔하5(ATR 기반 엔벨로프 밴드) 터치 시점을 찾아서, 터치 전후 5봉 동안
가격 기울기가 얼마나 "꺾였는지"(반전각도)를 계산하는 함수 모음이다.

배경(왜 이 파일이 필요한가)
--------------------------
기존 src/ma50_metrics.py, src/ma50_wave.py는 "MA20/MA50 이격도"를 기준으로
극값/파동을 찾았다. 이번 분석은 기준이 다르다 - 단순 이동평균이 아니라
"ATR(변동성)만큼 위아래로 벌어진 엔벨로프 밴드(엔상5/엔하5)"에 가격이 닿는
순간을 찾고, 그 순간 전후로 가격의 기울기(추세)가 실제로 반대 방향으로
바뀌는지를 "각도"로 측정한다.

엔상/엔하 공식 출처
-------------------
이 공식은 다른 프로젝트(C:\\Users\\HYEONJEONG\\stock-study)의
ichimoku_custom_indicator.py에서 이미 만들고 검증한 공식과 완전히 같은 구조다.
그 프로젝트는 이 프로젝트와 별개의 git 저장소/가상환경이라 코드를 직접
import하지 않고, 같은 계산식을 파라미터만 바꿔서(Period=50, ATRPeriod=20,
Factor=1.5) 이 프로젝트 스타일(한글 컬럼명, 상세 주석)에 맞게 새로 작성했다.

엔상(위쪽 밴드):
  엔상1 = MA(종가, 50)                                    <- 그냥 50일 이동평균
  엔상2 = 엔상1 + AVG(ATR(20), 10) x 1.5
  엔상3 = 엔상1 + AVG(ATR(20), 20) x 1.5 x 2
  엔상4 = 엔상1 + AVG(ATR(20), 20) x 1.5 x 3
  엔상5 = 엔상1 + AVG(ATR(20), 20) x 1.5 x 4
엔하(아래쪽 밴드)는 엔상과 완전히 대칭이다 (+ 대신 -).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 엔상/엔하 계산에 쓰는 파라미터 (사용자가 확정해준 값: Period=50, ATRPeriod=20,
# Factor=1.5). 나스닥 선물용(17,14,0.2)이나 삼성전자 참고용(17,14,1)과는
# 다른 값이므로, 이 파일 전용 상수로 따로 둔다.
# ---------------------------------------------------------------------------
MA_PERIOD = 50     # 엔상1(=중심선)에 쓰는 이동평균 기간
ATR_PERIOD = 20    # ATR(변동성 지표) 계산 기간
FACTOR = 1.5       # ATR 폭에 곱하는 배수

# 엔상2~5, 엔하2~5를 만들 때 쓰는 규칙.
# (몇 번째 선인지, ATR을 며칠치 평균낼지, 거기에 몇 배를 곱할지) 튜플의 목록.
# 2번선만 10일 평균 x1배, 3~5번선은 20일 평균 x(2,3,4)배 - 사용자가 검증해준
# 공식 그대로다.
LINE_SPECS = [
    (2, 10, 1),
    (3, 20, 2),
    (4, 20, 3),
    (5, 20, 4),
]

# "선3일목(10,20,50)" 참고용 파라미터 - 일목균형표 선행스팬 계산에 쓰는 기간들.
SPAN1_SHORT = 10
SPAN1_LONG = 20
SPAN2_PERIOD = 50

# "피보일목(5,8,13)" 참고용 파라미터 - 피보나치 수열 기반 단순이동평균 기간들.
FIBO_SHORT = 5
FIBO_MID = 8
FIBO_LONG = 13

# 터치 전/후로 기울기를 볼 봉의 개수 (요청 파라미터: N=5)
DEFAULT_N_BARS = 5


def add_atr(df, period=ATR_PERIOD):
    """
    ATR(Average True Range, 평균 실질 변동폭)을 계산해서 "ATR" 컬럼으로 추가한다.
    True Range = 그날의 (고가-저가), |고가-전일종가|, |저가-전일종가| 중 최댓값.
    ATR = True Range를 최근 period일 동안 단순평균(rolling mean)한 값.
    (값이 클수록 "요즘 변동성이 크다"는 뜻)
    """
    df = df.copy()  # 원본을 건드리지 않기 위해 복사본을 만든다.

    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)  # 하루 전 종가 (shift(1) = 한 칸 뒤로 밀기)

    # pd.concat으로 세 가지 후보값을 나란히 놓고, axis=1(가로 방향)으로 max를
    # 구하면 "그날그날 셋 중 가장 큰 값"이 True Range가 된다.
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.rolling(window=period).mean()
    return df


def _envelope_line(ma, atr, avg_window, multiplier, sign):
    """
    엔상/엔하의 개별 선 하나를 계산하는 내부 함수.
    공식: 중심선(ma) +/- ATR을 avg_window일 평균낸 값 x FACTOR x multiplier
    sign=+1이면 위(엔상), sign=-1이면 아래(엔하) 방향이 된다.
    """
    band_width = atr.rolling(window=avg_window).mean() * FACTOR * multiplier
    return ma + sign * band_width


def add_envelope_bands(df, ma_period=MA_PERIOD):
    """
    엔상1~5(위쪽 밴드), 엔하2~5(아래쪽 밴드) 컬럼을 추가한다.
    (엔하1=후행스팬은 미래 데이터를 미리 당겨써야 해서 실시간 판단에 못 쓰는
    참고용 선이라 이번 분석에서는 만들지 않는다)

    "ATR" 컬럼이 아직 없으면 자동으로 add_atr()을 먼저 호출해서 만든다.
    """
    df = df.copy()

    if "ATR" not in df.columns:
        df = add_atr(df)

    # 엔상1 = 그냥 50일 이동평균 (밴드들의 중심선 역할)
    ma = df["Close"].rolling(window=ma_period).mean()
    df["엔상1"] = ma

    # 엔상2~5: 중심선 위로 ATR 폭만큼 벌어지는 선들
    for line_no, avg_window, multiplier in LINE_SPECS:
        df[f"엔상{line_no}"] = _envelope_line(ma, df["ATR"], avg_window, multiplier, sign=1)

    # 엔하2~5: 엔상과 완전히 대칭 (아래쪽으로 벌어짐)
    for line_no, avg_window, multiplier in LINE_SPECS:
        df[f"엔하{line_no}"] = _envelope_line(ma, df["ATR"], avg_window, multiplier, sign=-1)

    return df


def add_span_10_20_50(df):
    """
    "선3일목(10,20,50)" 참고용 컬럼을 추가한다 (이번 분석에서는 필터로 쓰지
    않고, 결과 CSV에 참고 정보로만 남겨둔다).

    선1 = (최근 10일 최고가 + 최근 10일 최저가 + 최근 20일 최고가 + 최근 20일 최저가) / 4
    선2 = (최근 50일 최고가 + 최근 50일 최저가) / 2
    구름상태 = 선1이 선2보다 위이면 "양운", 아래이면 "음운"
    (일목균형표에서 "구름대 위/아래"로 큰 추세 방향을 보는 것과 같은 개념이다)
    """
    df = df.copy()

    hh_short = df["High"].rolling(window=SPAN1_SHORT).max()
    ll_short = df["Low"].rolling(window=SPAN1_SHORT).min()
    hh_long = df["High"].rolling(window=SPAN1_LONG).max()
    ll_long = df["Low"].rolling(window=SPAN1_LONG).min()
    hh_span2 = df["High"].rolling(window=SPAN2_PERIOD).max()
    ll_span2 = df["Low"].rolling(window=SPAN2_PERIOD).min()

    df["선1"] = (hh_short + ll_short + hh_long + ll_long) / 4
    df["선2"] = (hh_span2 + ll_span2) / 2

    # np.where(조건, 참일때값, 거짓일때값) : 조건에 따라 값을 골라주는 함수.
    df["구름상태"] = np.where(df["선1"] > df["선2"], "양운", "음운")
    # 아직 선1/선2가 계산 안 된(워밍업) 구간은 상태도 비워둔다.
    df.loc[df["선1"].isna() | df["선2"].isna(), "구름상태"] = None

    return df


def add_fibo_5_8_13(df):
    """
    "피보일목(5,8,13)" 참고용 컬럼을 추가한다 (마찬가지로 필터가 아니라 참고용).

    MA5/MA8/MA13 = 각각 5일/8일/13일 종가의 단순 이동평균
    피보4 = (MA5 + MA8) / 2   (더 단기 쪽 중간값)
    피보5 = (MA8 + MA13) / 2  (더 장기 쪽 중간값)
    """
    df = df.copy()

    ma5 = df["Close"].rolling(window=FIBO_SHORT).mean()
    ma8 = df["Close"].rolling(window=FIBO_MID).mean()
    ma13 = df["Close"].rolling(window=FIBO_LONG).mean()

    df["피보4"] = (ma5 + ma8) / 2
    df["피보5"] = (ma8 + ma13) / 2

    return df


def _slope_angle_deg(close_window):
    """
    5개(또는 n개) 종가로 이루어진 구간 하나를 받아서,
    "%변화율 기준 회귀선 기울기"를 각도(도)로 바꿔 반환한다.

    왜 %변화율을 쓰는가: 삼성전자(수만원대)와 대한광통신(수백원대)처럼
    종목마다 가격 단위가 완전히 다르다. 원화 그대로 기울기를 재면 비싼
    종목일수록 기울기가 커 보이는 착시가 생긴다. 그래서 구간 첫 봉 가격을
    기준(0%)으로 삼아 %변화율로 바꾼 뒤에 기울기를 재면, 종목이 달라도
    "몇 %씩/봉 움직였는가"로 공정하게 비교할 수 있다.
    """
    first_price = close_window.iloc[0]
    pct_change = (close_window - first_price) / first_price * 100  # %변화율로 변환

    x = np.arange(len(close_window))  # 0, 1, 2, 3, 4 (봉 순번)

    # np.polyfit(x, y, 1) : x,y에 가장 잘 맞는 직선(1차식) y = 기울기*x + 절편을
    # 구해준다. 반환값 중 첫 번째가 기울기다.
    slope_pct_per_bar = np.polyfit(x, pct_change.values, 1)[0]

    # 기울기(%/봉)를 각도(도)로 바꾼다. arctan(기울기) = 라디안 각도,
    # np.degrees()로 사람이 보기 편한 "도" 단위로 바꿔준다.
    angle_deg = np.degrees(np.arctan(slope_pct_per_bar))

    return slope_pct_per_bar, angle_deg


def add_all_indicators(df):
    """
    add_envelope_bands -> add_span_10_20_50 -> add_fibo_5_8_13 순서로 한 번에
    계산해주는 편의 함수. (run 스크립트 여러 개에서 매번 3줄씩 반복하지 않도록
    모아둔 것 뿐이고, 각 함수의 계산 내용 자체는 그대로다)
    """
    df = add_envelope_bands(df)
    df = add_span_10_20_50(df)
    df = add_fibo_5_8_13(df)
    return df


def find_band_touch_events(df, n_bars=DEFAULT_N_BARS, price_col="Close", band_no=5):
    """
    엔상N/엔하N 밴드 터치 이벤트를 찾아서, 터치 전후 n_bars봉의 반전각도를
    계산한 이벤트 표(데이터프레임)를 반환한다.

    band_no : 몇 번선을 볼지 (2~5). 기본값 5는 지금까지 써온 엔상5/엔하5와
    똑같이 동작한다 - 엔하3/엔하4처럼 "더 안쪽(중심선에 더 가까운) 밴드"도
    같은 로직으로 그대로 검증해보기 위해 파라미터로 뺐다.

    df는 add_envelope_bands(), add_span_10_20_50(), add_fibo_5_8_13()을
    모두 거친 상태여야 한다 (엔상N/엔하N/선1/선2/피보4/피보5 컬럼이 필요).

    터치 판정 기준(사용자 확정): 고가/저가 기준.
      - 상단터치: 그날 고가(High)가 엔상N 이상으로 올라간 경우
      - 하단터치: 그날 저가(Low)가 엔하N 이하로 내려간 경우
    """
    upper_col = f"엔상{band_no}"
    lower_col = f"엔하{band_no}"

    # 계산에 필요한 컬럼이 비어있는(워밍업 구간) 행은 제외한다.
    required_cols = [upper_col, lower_col, "High", "Low", price_col]
    valid = df.dropna(subset=required_cols).copy()
    if valid.empty:
        return pd.DataFrame()

    upper_touch = valid["High"] >= valid[upper_col]
    lower_touch = valid["Low"] <= valid[lower_col]

    records = []

    # 위쪽 터치, 아래쪽 터치를 각각 따로 처리한다 (같은 봉에서 둘 다 터치되는
    # 극히 드문 경우도 각자 독립된 이벤트로 취급하기 위해 분리한다).
    for touch_flag, band_label in [(upper_touch, f"엔상{band_no}터치"), (lower_touch, f"엔하{band_no}터치")]:
        if not touch_flag.any():
            continue

        # (조건이 바로 앞 행과 달라지는 지점마다 그룹 번호 +1) 패턴.
        # src/ma50_metrics.py의 find_extrema_events(), src/ma50_wave.py의
        # tag_waves()에서 이미 쓰던 것과 똑같은 방식이다. 이렇게 하면 "터치 봉이
        # 3일 연속 이어졌을 때"도 한 번의 이벤트로만 잡히고, 3번 중복 집계되지
        # 않는다.
        group_id = (touch_flag != touch_flag.shift()).cumsum()

        # 실제로 터치 조건이 참(True)인 그룹들만 골라서, 그룹의 "첫 봉"을
        # 터치 시각으로 삼는다.
        touch_groups = valid[touch_flag].groupby(group_id[touch_flag])

        for _, group_rows in touch_groups:
            touch_idx = group_rows.index[0]
            touch_pos = valid.index.get_loc(touch_idx)

            # 터치 시점 기준 앞 n_bars봉, 뒤 n_bars봉이 모두 있어야 계산 가능.
            # (데이터 맨 앞/맨 끝 근처의 터치는 앞뒤가 모자라서 건너뛴다)
            if touch_pos < n_bars or touch_pos + n_bars >= len(valid):
                continue

            before_window = valid[price_col].iloc[touch_pos - n_bars: touch_pos]
            after_window = valid[price_col].iloc[touch_pos + 1: touch_pos + 1 + n_bars]

            before_slope, before_angle = _slope_angle_deg(before_window)
            after_slope, after_angle = _slope_angle_deg(after_window)
            reversal_angle = after_angle - before_angle

            row = valid.loc[touch_idx]
            is_upper = band_label.startswith("엔상")
            touch_price = row["High"] if is_upper else row["Low"]

            # 반전성공여부: "원래 가던 방향으로 계속 가다가, 터치 후 반대로
            # 꺾였는가"를 바로 볼 수 있는 컬럼.
            #   상단터치라면: 터치 전엔 오르는 중(전각도>0)이었는데, 터치 후
            #                실제로 각도가 내려가는 쪽으로 바뀌었는지(반전각도<0)
            #   하단터치는 정반대 방향으로 같은 논리를 적용한다.
            if is_upper:
                reverted = bool(before_angle > 0 and reversal_angle < 0)
            else:
                reverted = bool(before_angle < 0 and reversal_angle > 0)

            deviation_pct = (row[price_col] - row["엔상1"]) / row["엔상1"] * 100

            records.append(
                {
                    "터치시각": touch_idx,
                    "밴드종류": band_label,
                    "터치가격": touch_price,
                    "이격도(%)": deviation_pct,
                    "터치전기울기(%/봉)": before_slope,
                    "터치전각도(도)": before_angle,
                    "터치후기울기(%/봉)": after_slope,
                    "터치후각도(도)": after_angle,
                    "반전각도(도)": reversal_angle,
                    "반전성공여부": reverted,
                    "구름상태": row.get("구름상태"),
                    "피보4": row.get("피보4"),
                    "피보5": row.get("피보5"),
                }
            )

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).sort_values("터치시각").reset_index(drop=True)
    return result


def simulate_reversal_trades(df, events, n_bars=DEFAULT_N_BARS, price_col="Close", ma_col="엔상1"):
    """
    "밴드 터치 = 반전 신호"라고 보고 실제로 매매했다면 어땠을지 시뮬레이션한다.

    매매 규칙 (요청 그대로, 밴드 번호에 관계없이 동일하게 적용):
      - 엔상N 터치 -> 숏(매도) 진입, 엔하N 터치 -> 롱(매수) 진입
      - 진입가 = 터치 당일 종가 (고가/저가로 터치 여부는 이미 알 수 있고, 그날
        종가는 장 마감 시점에 이미 확정된 값이므로 미래 정보를 미리 아는 것이
        아니다)
      - 청산: 터치 다음 날부터 최대 n_bars(=5)봉 동안 지켜보다가,
          * 종가가 MA50(엔상1)까지 돌아오면 그 시점에 "MA50복귀"로 청산
          * n_bars봉이 지나도록 못 돌아오면 마지막 봉에서 "N봉경과"로 강제 청산
        (둘 중 먼저 오는 조건으로 청산 - 요청하신 "N봉 후 청산 또는 MA50
        복귀시 청산"을 그대로 구현한 것)

    df는 add_all_indicators()를 거친 종목 1개의 전체 일봉(엔상1/MA50 포함)이고,
    events는 find_band_touch_events()로 만든 그 종목의 터치 이벤트 표다.
    반환값은 거래 1건당 1행인 데이터프레임이다 (events의 컨텍스트 컬럼 + 매매
    결과 컬럼).
    """
    trades = []

    for _, ev in events.iterrows():
        touch_idx = ev["터치시각"]
        touch_pos = df.index.get_loc(touch_idx)

        direction = "숏" if ev["밴드종류"].startswith("엔상") else "롱"
        entry_price = df[price_col].iloc[touch_pos]

        exit_pos = None
        exit_reason = None

        # 터치 다음날부터 최대 n_bars봉까지 하루씩 확인한다.
        for step in range(1, n_bars + 1):
            candidate_pos = touch_pos + step
            if candidate_pos >= len(df):
                break  # 데이터가 여기서 끝나버린 경우 (이론상 events 생성 단계에서 걸러지지만 안전장치)

            close_i = df[price_col].iloc[candidate_pos]
            ma_i = df[ma_col].iloc[candidate_pos]

            if direction == "숏" and close_i <= ma_i:
                exit_pos, exit_reason = candidate_pos, "MA50복귀"
                break
            if direction == "롱" and close_i >= ma_i:
                exit_pos, exit_reason = candidate_pos, "MA50복귀"
                break

        if exit_pos is None:
            # n_bars봉 안에 MA50으로 못 돌아온 경우, n_bars번째 봉에서 강제 청산.
            exit_pos = min(touch_pos + n_bars, len(df) - 1)
            exit_reason = f"{n_bars}봉경과"

        exit_price = df[price_col].iloc[exit_pos]
        hold_bars = exit_pos - touch_pos

        if direction == "숏":
            pnl_pct = (entry_price - exit_price) / entry_price * 100  # 숏은 가격이 내려가야 이익
        else:
            pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append(
            {
                "터치시각": touch_idx,
                "밴드종류": ev["밴드종류"],
                "방향": direction,
                "진입가": entry_price,
                "청산가": exit_price,
                "보유봉수": hold_bars,
                "청산사유": exit_reason,
                "손익률(%)": pnl_pct,
                "반전각도(도)": ev["반전각도(도)"],
                "반전성공여부": ev["반전성공여부"],
            }
        )

    return pd.DataFrame(trades)


def simulate_target_exit_trades(df, events, exit_band_no, price_col="Close"):
    """
    엔상N을 "숏 진입 신호"가 아니라 "롱 포지션의 청산 목표가"로 재해석해서
    시뮬레이션한다. (엔하 매수 진입은 이미 만들어둔 events를 그대로 쓰고,
    청산 규칙만 이 함수가 다르게 적용한다)

    청산 규칙: 진입일 다음 날부터, 고가(High)가 엔상{exit_band_no} 이상으로
    올라간 "첫" 날의 종가에 청산한다. simulate_reversal_trades()와 달리
    n_bars 같은 보유기간 상한이 없다 - 목표가에 닿을 때까지 계속 들고 있는다.
    데이터가 끝날 때까지 한 번도 못 닿으면, 마지막 봉 종가로 강제 청산하고
    청산사유를 "미도달(데이터종료)"로 남긴다 (ma50_backtest.py의 "강제청산"과
    같은 개념).

    df는 add_all_indicators()를 거친 종목 1개의 전체 일봉이고, events는
    (다른 진입 밴드로) find_band_touch_events()가 만든 그 종목의 매수 진입
    이벤트 표다. 진입가는 simulate_reversal_trades()와 똑같이 "진입일 종가"를
    쓴다 (같은 날 고가로 이미 밴드를 찍었더라도, 미래 정보를 미리 안 쓰기
    위해 그날 종가로 체결됐다고 가정하는 이 프로젝트의 기존 방식과 통일).
    """
    exit_col = f"엔상{exit_band_no}"
    exit_reason_label = f"엔상{exit_band_no}터치"
    trades = []

    for _, ev in events.iterrows():
        touch_idx = ev["터치시각"]
        touch_pos = df.index.get_loc(touch_idx)
        entry_price = df[price_col].iloc[touch_pos]

        exit_pos = None
        exit_reason = None

        for candidate_pos in range(touch_pos + 1, len(df)):
            if df["High"].iloc[candidate_pos] >= df[exit_col].iloc[candidate_pos]:
                exit_pos = candidate_pos
                exit_reason = exit_reason_label
                break

        if exit_pos is None:
            # 데이터 끝까지 목표가에 못 닿은 경우, 마지막 봉 종가로 강제 청산.
            exit_pos = len(df) - 1
            exit_reason = "미도달(데이터종료)"

        exit_price = df[price_col].iloc[exit_pos]
        hold_bars = exit_pos - touch_pos
        pnl_pct = (exit_price - entry_price) / entry_price * 100  # 전부 롱이므로 이 방향 그대로

        trades.append(
            {
                "터치시각": touch_idx,
                "청산방식": f"엔상{exit_band_no}청산",
                "진입가": entry_price,
                "청산가": exit_price,
                "보유봉수": hold_bars,
                "청산사유": exit_reason,
                "손익률(%)": pnl_pct,
            }
        )

    return pd.DataFrame(trades)


def build_entry_events(df, entry_band, n_bars=DEFAULT_N_BARS, train_ratio=0.7):
    """
    엔하{entry_band}터치를 매수 진입 이벤트로 뽑고, 학습/검증 구간 라벨까지
    붙여서 반환한다.

    run_enha_entry_ensah_exit_analysis.py에서 처음 이 패턴을 썼고, 이후
    스크립트에서도 계속 반복해서 쓰는 "엔하N 진입 이벤트 만들기" 절차라서
    공용 함수로 뺐다 (기존 스크립트의 로직과 완전히 동일).
    """
    from src.ma50_backtest import split_train_test  # 지연 import로 순환참조 방지

    band_label = f"엔하{entry_band}터치"
    events = find_band_touch_events(df, n_bars=n_bars, band_no=entry_band)
    if events.empty:
        return pd.DataFrame()

    events = events[events["밴드종류"] == band_label].copy()
    if events.empty:
        return pd.DataFrame()

    train_part, _ = split_train_test(df, train_ratio=train_ratio)
    split_time = train_part.index[-1] if len(train_part) > 0 else df.index[-1]
    events["구간"] = np.where(events["터치시각"] <= split_time, "학습", "검증")

    return events
