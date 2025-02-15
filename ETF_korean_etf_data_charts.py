import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 한글 폰트 설정 (Windows: 'Malgun Gothic', macOS: 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 사용자
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호 깨짐 방지

# ✅ 한국 ETF 9종목 리스트
korean_etfs = ["428510", "456250", "456600", "465660", "463050", "466920", "475080", "478150", "486450"]

# ✅ ETF 티커 → 한국어 이름 매핑
etf_names = {
    "428510": "차이나메타버스",
    "456250": "유럽명품TOP10",
    "456600": "글로벌AI",
    "465660": "일본반도체",
    "463050": "K바이오",
    "466920": "조선TOP3",
    "475080": "테슬라커버드콜",
    "478150": "글로벌우주&방산",
    "486450": "미국AI전력인프라"
}

# ✅ 모든 ETF 데이터를 불러와서 하나의 데이터프레임으로 병합
df_list = []
for ticker in korean_etfs:
    filename = f"etf_{ticker}_2024-07-16_2025-02-07.csv"  # 최신 크롤링된 파일 사용
    df = pd.read_csv(filename, usecols=["날짜", "종가"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df.set_index("날짜", inplace=True)
    df.rename(columns={"종가": ticker}, inplace=True)
    df_list.append(df)

# ✅ 하나의 데이터프레임으로 병합
df_prices = pd.concat(df_list, axis=1).dropna()  # 공통된 날짜만 유지
df_prices.sort_index(inplace=True)

# ✅ 데이터 확인
print("📌 ETF 종가 데이터")
print(df_prices.head())

# ✅ 기준일 (2024-07-16) 가격으로 정규화
df_normalized = df_prices / df_prices.iloc[0] * 100  # 첫 번째 항목을 100으로 정규화

# ✅ 한국 이름 적용 (컬럼명 변경)
df_prices.rename(columns=etf_names, inplace=True)
df_normalized.rename(columns=etf_names, inplace=True)

# ✅ 첫번째 그래프: ETF 가격 변동 차트 (한국 이름 적용)
df_prices.plot(figsize=(12,6), title="ETF 가격 변동 (종가)")
plt.xlabel("날짜")
plt.ylabel("가격 (원)")
plt.legend(title="ETF", loc='upper left')  # 🔥 범례를 왼쪽 상단으로 이동
plt.grid(True)

# ✅ 두번째 그래프: Normalized Price 차트 (한국 이름 적용)
df_normalized.plot(figsize=(12,6), title="ETF 수익률 비교 (2024-07-16 = 100)")
plt.xlabel("날짜")
plt.ylabel("Normalized Price (기준: 2024-07-16 = 100)")
plt.legend(title="ETF", loc='upper left')  # 🔥 범례를 왼쪽 상단으로 이동
plt.grid(True)

# ✅ 모든 그래프 출력 (plt.show()는 마지막에 한 번만 호출)
plt.show()