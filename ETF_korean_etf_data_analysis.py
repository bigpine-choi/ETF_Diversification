import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 한글 폰트 설정 (Windows: 'Malgun Gothic', macOS: 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ ETF 14종목 리스트
korean_etfs = ["428510", "456250", "456600", "465660", "463050", "466920", "478150", "486450", "475050", "468380",
               "153130", "475080", "481340", "434060"]

# ✅ ETF 티커 → 한국어 이름 매핑
etf_names = {
    "428510": "차이나메타버스",
    "456250": "유럽명품TOP10",
    "456600": "글로벌AI",
    "465660": "일본반도체",
    "463050": "K바이오",
    "466920": "조선TOP3",
    "478150": "글로벌우주&방산",
    "486450": "미국AI전력인프라",
    "475050": "KPOP포커스",
    "468380": "미국하이일드",
    "153130": "단기채권(현금)",
    "475080": "테슬라커버드콜",
    "481340": "미국채30년",
    "434060": "TDF2050",
}

# ✅ 모든 ETF 데이터를 불러와서 하나의 데이터프레임으로 병합
df_list = []
for ticker in korean_etfs:
    filename = f"etf_{ticker}_2024-07-16_2025-03-14.csv"  # 최신 크롤링된 파일 사용
    df = pd.read_csv(filename, usecols=["날짜", "종가"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df.set_index("날짜", inplace=True)
    df.rename(columns={"종가": ticker}, inplace=True)
    df_list.append(df)

# ✅ 하나의 데이터프레임으로 병합
df_prices = pd.concat(df_list, axis=1).dropna()  # 공통된 날짜만 유지
df_prices.sort_index(inplace=True)

# ✅ 수익률(Returns) 계산
df_returns = df_prices.pct_change().dropna()

# ✅ 변동성(Volatility) 계산 (30일 이동 표준편차)
df_volatility = df_returns.rolling(window=30).std()

# ✅ 상관관계(Correlation) 계산
df_correlation = df_returns.corr()

# ✅ 한국 이름 적용 (컬럼명 변경)
df_prices.rename(columns=etf_names, inplace=True)
df_returns.rename(columns=etf_names, inplace=True)
df_volatility.rename(columns=etf_names, inplace=True)
df_correlation.rename(index=etf_names, columns=etf_names, inplace=True)

# ✅ 첫번째 그래프: ETF 수익률 변동 차트
df_returns.plot(figsize=(12,6), title="ETF 일별 수익률 변동")
plt.xlabel("날짜")
plt.ylabel("수익률")
plt.legend(title="ETF", loc='upper left')
plt.grid(True)

# ✅ 두번째 그래프: 변동성 차트 (30일 이동 표준편차)
df_volatility.plot(figsize=(12,6), title="ETF 변동성 (30일 이동 표준편차)")
plt.xlabel("날짜")
plt.ylabel("변동성 (표준편차)")
plt.legend(title="ETF", loc='upper left')
plt.grid(True)

# ✅ 세번째 그래프: 상관관계 히트맵
plt.figure(figsize=(12,8)) # 그림 크기 조정
sns.heatmap(df_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 10})
plt.xticks(rotation=0, fontsize=6)

plt.title("ETF 간 상관관계", fontsize=14) # 제목 크기 조정
plt.show()