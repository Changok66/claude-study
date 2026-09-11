# -*- coding: utf-8 -*-
"""
MA50 이격도 분석에 필요한 원본 시세 데이터를 받아오는(다운로드) 함수 모음이다.

yfinance : 야후 파이낸스에서 주가/선물 시세를 무료로 받아오는 라이브러리다.
           이 프로젝트에는 아직 실제 시세 데이터가 없고, 나스닥 선물(NQ=F)과
           국내 주식을 동시에 무료로 받을 수 있는 사실상 표준 라이브러리라서 선택했다.
           (pip install yfinance 로 .venv 안에 설치해야 사용할 수 있다)
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

# -----------------------------------------------------------------------------
# 국내 보유종목 9개의 "종목코드"만 모아둔 표.
# 야후 파이낸스에서 한국 주식은 종목코드 뒤에 거래소를 뜻하는 접미사를 붙여야 한다.
#   .KS = 코스피(KOSPI), .KQ = 코스닥(KOSDAQ)
# 아래 코드는 기억을 바탕으로 적어둔 값이라 100% 확신할 수 없으므로,
# resolve_kr_symbol() 함수가 .KS를 먼저 시도해보고 안 되면 .KQ로 다시 시도한다.
# -----------------------------------------------------------------------------
KR_STOCK_CODES = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "두산에너빌리티": "034020",
    "테크윙": "089030",
    "아모레퍼시픽": "090430",
    "HD현대중공업": "329180",
    "HD현대일렉트릭": "267260",
    "에이피알": "278470",
    "대한광통신": "010170",
}


def resolve_kr_symbol(code, min_rows=200):
    """
    국내 종목코드(예: "005930")에 .KS 또는 .KQ를 붙여서
    야후 파이낸스가 실제로 데이터를 주는 "완성된 티커"를 찾아주는 함수.

    주의: 야후 파이낸스는 종목코드가 틀려도(혹은 상장폐지된 옛 종목이어도)
    완전히 빈 데이터가 아니라 아주 짧은(며칠치) 데이터를 돌려줄 때가 있다.
    그래서 단순히 "비어있지 않으면 통과"가 아니라, "데이터가 min_rows개
    이상일 때만 그 티커가 맞다고 인정"하도록 검사한다. (실제로 이 검사가
    없어서 테크윙/대한광통신이 .KS로 잘못 매칭된 적이 있었다)

    반환값: (완성된 티커 문자열, 일봉 데이터프레임)
            둘 다 못 찾으면 (None, 빈 데이터프레임)
    """
    for suffix in [".KS", ".KQ"]:
        symbol = code + suffix

        # yf.Ticker(...).history(...) : 그 종목의 과거 시세를 받아오는 함수.
        # period="max" : 상장일부터 지금까지 가능한 한 가장 긴 기간을 받는다.
        # interval="1d" : 하루 단위(일봉)로 받는다.
        df = yf.Ticker(symbol).history(period="max", interval="1d")

        # 데이터가 min_rows행 이상이어야 "진짜 그 종목"으로 인정한다.
        if len(df) >= min_rows:
            return symbol, df

    # .KS, .KQ 둘 다 실패한 경우
    return None, pd.DataFrame()


def fetch_kr_daily_bars(name):
    """
    국내 보유종목 이름(예: "삼성전자")을 받아서 일봉 데이터프레임을 반환한다.
    KR_STOCK_CODES 표에 없는 이름이면 에러를 낸다.
    """
    if name not in KR_STOCK_CODES:
        raise ValueError(f"'{name}'은(는) KR_STOCK_CODES 표에 없는 종목명입니다.")

    code = KR_STOCK_CODES[name]
    symbol, df = resolve_kr_symbol(code)
    return symbol, df


def fetch_nq_4min_bars(max_days=30, chunk_days=7):
    """
    나스닥100 E-mini 선물(NQ=F)의 "진짜 4분봉"을 만들어서 반환한다.

    문제: 야후 파이낸스(yfinance)는 4분봉을 직접 제공하지 않는다.
          (지원하는 간격은 1분,2분,5분,15분,30분,60분,90분,1일 ... 뿐이다)
    해결: 가장 잘게 쪼갠 1분봉을 받은 뒤, pandas의 resample() 기능으로
          "1분봉 4개를 묶어서 4분봉 1개"로 직접 합쳐서(리샘플링) 만든다.

    추가 제약: 야후 파이낸스는 1분봉을 "최근 약 30일치"까지만 제공하고,
               한 번 요청할 때 최대 7일 범위까지만 허용한다.
               그래서 최근 시점부터 7일씩 거슬러 올라가며 여러 번 나눠받고
               (chunk_days=7), 전체적으로는 max_days(기본 30일)까지만 모은다.
    """
    ticker = yf.Ticker("NQ=F")

    # 지금 시각(UTC 기준)을 기준으로, 얼마나 과거까지 데이터를 모을지 범위를 정한다.
    end_overall = datetime.now(timezone.utc)
    start_overall = end_overall - timedelta(days=max_days)
    chunk = timedelta(days=chunk_days)

    frames = []  # 7일치씩 받아온 데이터프레임들을 여기에 쌓아둔다.
    cur_end = end_overall

    # cur_end가 전체 시작점(start_overall)보다 뒤에 있는 동안 계속 반복해서
    # 7일씩 거슬러 올라가며 데이터를 받는다.
    while cur_end > start_overall:
        # max(...)를 쓰는 이유: 마지막 조각은 7일보다 짧게 남을 수 있기 때문에
        # start_overall보다 더 과거로 넘어가지 않도록 잘라준다.
        cur_start = max(cur_end - chunk, start_overall)

        hist = ticker.history(start=cur_start, end=cur_end, interval="1m")
        if not hist.empty:
            frames.append(hist)

        cur_end = cur_start  # 다음 반복에서는 이번 구간의 시작점부터 다시 거슬러 올라간다.

    if not frames:
        # 데이터를 하나도 못 받은 경우 (주말/휴장 등으로 아예 없을 수도 있음)
        return pd.DataFrame()

    # 여러 조각을 하나로 합친다. (pd.concat = 데이터프레임 여러 개를 이어붙이기)
    merged = pd.concat(frames)

    # 구간이 겹쳐서 같은 시각의 봉이 중복으로 들어있을 수 있으므로 제거한다.
    merged = merged[~merged.index.duplicated(keep="first")]

    # 시간 순서대로 정렬한다. (과거 -> 최근)
    merged = merged.sort_index()

    # ---- 1분봉 -> 4분봉으로 리샘플링 ----
    # resample("4min") : 시간 인덱스를 4분 단위 구간으로 다시 묶는다.
    # label="left", closed="left" : 구간의 "시작 시각"을 그 봉의 시각으로 표시한다.
    # OHLCV(시가/고가/저가/종가/거래량) 각각을 올바른 방식으로 합친다.
    #   시가(Open)  = 구간 내 "첫" 1분봉의 시가
    #   고가(High)  = 구간 내 "최댓" 고가
    #   저가(Low)   = 구간 내 "최솟" 저가
    #   종가(Close) = 구간 내 "마지막" 1분봉의 종가
    #   거래량(Volume) = 구간 내 거래량 "합계"
    four_min = merged.resample("4min", label="left", closed="left").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    # 실제 거래가 없어서 텅 빈 4분 구간(Close가 NaN)은 분석에 방해가 되므로 제거한다.
    four_min = four_min.dropna(subset=["Close"])

    return four_min
