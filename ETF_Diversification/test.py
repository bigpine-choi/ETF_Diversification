import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

# ✅ 한글 폰트 설정 (Windows: 'Malgun Gothic', macOS: 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ ETF 9종목 리스트
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

# ✅ ETF 종가 데이터 로드 & 병합
df_list = []
for ticker in korean_etfs:
    filename = f"etf_{ticker}_2024-07-16_2025-02-07.csv"
    df = pd.read_csv(filename, usecols=["날짜", "종가"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df.set_index("날짜", inplace=True)
    df.rename(columns={"종가": etf_names[ticker]}, inplace=True)
    df_list.append(df)

df_prices = pd.concat(df_list, axis=1).dropna()
df_prices.sort_index(inplace=True)

# ✅ 수익률(Returns) 계산
df_returns = df_prices.pct_change().dropna()

# ✅ 기대 수익률과 공분산 행렬 계산
mean_returns = df_returns.mean()
cov_matrix = df_returns.cov()

# ✅ 실제 시장 금리 기반 무위험 수익률 적용 (예: 한국은행 기준금리 활용)
risk_free_rate = 0.03/252 # 0.0119% (향후 실제 크롤링한 시장 데이터로 업데이트 가능)

# ✅ 랜덤 포트폴리오 샘플링
num_portfolios = 100000
results = np.zeros((4, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(korean_etfs)), size=1)[0]  # 랜덤 가중치 합 = 1
    expected_return = np.sum(weights * mean_returns)
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility  # ✅ 무위험 수익률 반영

    results[0, i] = expected_return
    results[1, i] = expected_volatility
    results[2, i] = sharpe_ratio
    results[3, i] = i  # 인덱스 저장
    weights_record.append(weights)

# ✅ 샤프 비율 최대화 포트폴리오 찾기
max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]

# ✅ 최적 포트폴리오 정보
optimal_return = results[0, max_sharpe_idx]  # 기대수익률
optimal_volatility = results[1, max_sharpe_idx]  # 변동성
optimal_sharpe = results[2, max_sharpe_idx]  # ✅ 수정된 샤프 비율

# ✅ 효율적 투자선 계산
def min_variance(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 가중치 합 = 1
bounds = tuple((0, 1) for _ in range(len(korean_etfs)))  # 가중치 범위: 0 ~ 1

efficient_portfolio = []
target_returns = np.linspace(results[0].min(), results[0].max(), 100)

for target in target_returns:
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target}
    )
    result = sco.minimize(min_variance, len(korean_etfs) * [1. / len(korean_etfs)],
                          method='SLSQP', bounds=bounds, constraints=constraints)
    efficient_portfolio.append(result.fun)

# ✅ 시각화
fig, ax = plt.subplots(figsize=(12, 6))

# ✅ 랜덤 포트폴리오 & 최적 포트폴리오 & 효율적 투자선
scatter = ax.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5, label="랜덤 포트폴리오")
ax.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], color='red', marker='*', s=200, label="최적 포트폴리오")
ax.plot(efficient_portfolio, target_returns, color='black', linestyle='--', label="효율적 투자선")  # ✅ 효율적 투자선 복원

# ✅ 최적 포트폴리오 정보 박스
text = (f"최적 포트폴리오\n"
        f"기대 수익률: {optimal_return:.2%}\n"
        f"변동성: {optimal_volatility:.2%}\n"
        f"샤프 비율: {optimal_sharpe:.2f}")

ax.text(0.014, 0.0016, text,  # ✅ 위치 수정
        fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# ✅ 차트 내부에 ETF 비중을 표로 삽입
table_data = [[name, f"{weight:.1%}"] for weight, name in zip(optimal_weights, etf_names.values())]

table = plt.table(cellText=table_data, colLabels=["ETF", "비중"],
                  cellLoc='center', loc='right', bbox=[1.2, 0.2, 0.3, 0.6])

table.auto_set_font_size(False)
table.set_fontsize(9)  # ✅ 폰트 크기 조정

# ✅ 오른쪽 여백 확보
fig.subplots_adjust(right=0.85)

ax.set_xlabel("리스크 (표준편차)")
ax.set_ylabel("기대 수익률")
ax.set_title("포트폴리오 최적화 (효율적 투자선 & 샤프 비율 최적화)")
ax.legend()
fig.colorbar(scatter, label="샤프 비율")
ax.grid(True)
plt.show()

# ✅ 최적 포트폴리오 출력
optimal_portfolio = pd.DataFrame({'ETF': list(etf_names.values()), '비중': optimal_weights})
print("\n📌 최적 포트폴리오 구성 (샤프 비율 최대화):")
print(optimal_portfolio)