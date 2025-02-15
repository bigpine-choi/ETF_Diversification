import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random
from concurrent.futures import ThreadPoolExecutor

# ✅ 한국 ETF 9종목
korean_etfs = ["428510", "456250", "456600", "465660", "463050", "466920", "475080", "478150", "486450"]

# ✅ 네이버 금융 일별 시세 URL
BASE_URL = "https://finance.naver.com/item/sise_day.nhn?code={}&page={}"

# ✅ 원하는 날짜 범위 설정
start_date = "2024-07-16"
end_date = "2025-02-16"


# ✅ 최대 페이지 수 가져오는 함수 (더 정확한 최대 페이지 탐색)
def get_max_pages(ticker, session):
    url = BASE_URL.format(ticker, 1)
    headers = {"User-Agent": "Mozilla/5.0"}
    response = session.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    # 페이지 네이션 영역에서 마지막 페이지 번호 찾기
    pagination = soup.select("td.pgRR a")
    if pagination:
        try:
            last_page = int(pagination[0]['href'].split('=')[-1])
        except:
            last_page = 20  # 기본 20페이지 크롤링
    else:
        last_page = 20  # 기본 20페이지 크롤링

    # ✅ 만약 페이지가 적다면 더 깊이 확인
    if last_page < 50:  # 50페이지 이하라면 추가 검사
        for page in range(last_page, last_page + 5):
            url = BASE_URL.format(ticker, page)
            response = session.get(url, headers=headers)
            if "일별 시세가 없습니다" in response.text:
                break
            last_page = page

    return last_page


# ✅ ETF 크롤링 함수 (최대 페이지 적용 & 빠른 연결 사용)
def fetch_etf_data(ticker):
    all_data = []

    with requests.Session() as session:  # 세션 사용하여 연결 재사용
        max_pages = get_max_pages(ticker, session)  # 최대 페이지 수 확인

        for page in range(1, max_pages + 1):  # 동적으로 결정된 페이지만 크롤링
            url = BASE_URL.format(ticker, page)
            headers = {"User-Agent": "Mozilla/5.0"}
            response = session.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'lxml')

            # 테이블 데이터 가져오기
            table = soup.find("table", class_="type2")
            rows = table.find_all("tr")[2:]  # 첫 번째 줄은 컬럼명이므로 제외

            if not rows:  # 더 이상 데이터가 없으면 종료
                break

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 6:
                    continue  # 빈 행 스킵

                # 날짜, 종가, 거래량 가져오기
                date = cols[0].text.strip()
                close_price = cols[1].text.strip().replace(",", "")
                volume = cols[6].text.strip().replace(",", "")

                if close_price and volume:
                    all_data.append([date, float(close_price), int(volume)])

            time.sleep(random.uniform(0.05, 0.2))  # ✅ 대기 시간 최소화 (0.05~0.2초)

    # ✅ DataFrame 변환 & 날짜 필터링 적용
    df = pd.DataFrame(all_data, columns=["날짜", "종가", "거래량"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values(by="날짜")

    # ✅ 중복된 날짜 제거 (모든 ETF에 적용)
    df = df.drop_duplicates(subset=["날짜"], keep="first")  # ✅ 모든 ETF에서 중복된 날짜 제거
    print(f"⚠️ {ticker} ETF에서 중복된 날짜 제거 완료! (중복 제거 후 {len(df)}개 데이터)")

    # ✅ 사용자 지정 날짜 범위만 필터링
    df = df[(df["날짜"] >= start_date) & (df["날짜"] <= end_date)]

    # ✅ 크롤링된 실제 데이터의 시작일 & 종료일 확인
    if not df.empty:
        actual_start_date = max(df["날짜"].min(), pd.to_datetime(start_date)).strftime("%Y-%m-%d")  # ✅ 수정된 부분
        actual_end_date = min(df["날짜"].max(), pd.to_datetime(end_date)).strftime("%Y-%m-%d")  # ✅ 종료일도 동일한 방식 적용
    else:
        actual_start_date = start_date  # 데이터가 없을 경우 기존 start_date 유지
        actual_end_date = end_date  # 데이터가 없을 경우 기존 end_date 유지

    # ✅ 파일명에 실제 시작일 & 종료일 반영
    filename = f"etf_{ticker}_{actual_start_date}_{actual_end_date}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"✅ {ticker} ETF 데이터 저장 완료 ({len(df)}개 데이터) → {filename}")


# ✅ 병렬 크롤링 실행 (ThreadPoolExecutor 사용, max_workers 증가)
with ThreadPoolExecutor(max_workers=8) as executor:  # ✅ 8개 스레드로 속도 향상
    executor.map(fetch_etf_data, korean_etfs)