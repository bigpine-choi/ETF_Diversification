import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

# ✅ 한글 폰트 설정 (Windows: 'Malgun Gothic')
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

# ✅ '안전자산'과 '위험자산' 구분
safe_assets = ["153130", "475080", "481340", "434060"]  # 단기채권(현금), 테슬라커버드콜, 미국채30년, TDF2050
risky_assets = [ticker for ticker in korean_etfs if ticker not in safe_assets]

# ✅ ETF 종가 데이터 로드 & 병합
df_list = []
for ticker in korean_etfs:
    filename = f"etf_{ticker}_2024-07-16_2025-03-14.csv"
    df = pd.read_csv(filename, usecols=["날짜", "종가"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df.set_index("날짜", inplace=True)
    df.rename(columns={"종가": etf_names[ticker]}, inplace=True)
    df_list.append(df)

df_prices = pd.concat(df_list, axis=1).dropna()
df_prices.sort_index(inplace=True)

# ✅ 수익률(Returns) 계산
df_returns = df_prices.pct_change().dropna()

# ✅ 기대 수익률과 공분산 행렬 계산 (연환산)
trading_days = 252
mean_daily_returns = np.exp(np.log(1 + df_returns).mean()) - 1
mean_annual_returns = (1 + mean_daily_returns) ** trading_days - 1
annual_cov_matrix = np.nan_to_num(df_returns.cov() * trading_days, nan=0.0)

# ✅ 무위험 수익률 (3%)
risk_free_rate = 0.03

# ✅ 랜덤 포트폴리오 생성
num_portfolios = 300000
results = np.zeros((4, num_portfolios))
weights_record = []  # 리스트로 유지

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(korean_etfs)), size=1)[0]
    expected_return = np.sum(weights * mean_annual_returns)
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
    sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility

    if not np.isnan(expected_return) and not np.isnan(expected_volatility):
        results[0, i] = expected_return
        results[1, i] = expected_volatility
        results[2, i] = sharpe_ratio
        results[3, i] = i
        weights_record.append(weights)  # ✅ 리스트로 추가

# ✅ 모든 데이터를 추가한 후 NumPy 배열로 변환
weights_record = np.array(weights_record)  # ✅ 최적화 후 NumPy 배열로 변환

# ✅ 최적 포트폴리오 찾기 (샤프 비율 최대화)
max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]  # ✅ 비중
optimal_return = results[0, max_sharpe_idx]  # 기대수익률
optimal_volatility = results[1, max_sharpe_idx]  # 변동성
optimal_sharpe = results[2, max_sharpe_idx]  # ✅ 수정된 샤프 비율


# ✅ 효율적 투자선 계산
def min_variance(weights):
    return np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))  # ✅ annual_cov_matrix


constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # ✅ 가중치 합 = 1
bounds = tuple((0, 1) for _ in range(len(korean_etfs)))  # ✅ 가중치 범위: 0 ~ 1

efficient_portfolio = []
target_returns = np.linspace(results[0].min(), results[0].max(), 100)

for target in target_returns:
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(mean_annual_returns * x) - target}  # ✅ mean_annual_returns
    )
    # ✅ 최적화 반복 제한 추가
    result = sco.minimize(
        min_variance,
        len(korean_etfs) * [1. / len(korean_etfs)],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500}  # ✅ 반복 제한 추가
    )
    efficient_portfolio.append(result.fun if result.success else np.nan)  # ✅ 실패 시 기본값 적용

# ✅ 그래프 생성
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5, label="랜덤 포트폴리오")
ax.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], color='red', marker='*', s=200, label="최적 포트폴리오")
ax.plot(efficient_portfolio, target_returns, color='black', linestyle='--', label="효율적 투자선")

# ✅ 최적 포트폴리오 정보 박스
text = (f"최적 포트폴리오\n"
        f"기대 수익률: {optimal_return:.2%}\n"
        f"변동성: {optimal_volatility:.2%}\n"
        f"샤프 비율: {optimal_sharpe:.2f}")
ax.text(0.15, 0.3, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# ✅ 차트 내부에 ETF 비중을 표로 삽입
table_data = [[name, f"{weight:.1%}"] for weight, name in zip(optimal_weights, etf_names.values())]
table = plt.table(cellText=table_data, colLabels=["ETF", "비중"],
                  cellLoc='center', loc='right', bbox=[1.2, 0.2, 0.3, 0.6])  # ✅ 위치 조정
table.auto_set_font_size(False)
table.set_fontsize(9)  # ✅ 폰트 크기 조정
fig.subplots_adjust(right=0.85)  # ✅ 오른쪽 여백 확보

# ✅ 범례 표시
ax.set_xlabel("리스크 (표준편차)")
ax.set_ylabel("기대 수익률")
ax.set_title("포트폴리오 최적화 (효율적 투자선 & 샤프 비율 최적화)")
ax.legend()
fig.colorbar(scatter, label="샤프 비율")
ax.grid(True)
plt.show()

# ✅ 최적 포트폴리오 비중 출력(콘솔)
optimal_portfolio = pd.DataFrame({'ETF': list(etf_names.values()), '비중': optimal_weights})
optimal_portfolio["비중"] = optimal_portfolio["비중"].map(lambda x: f"{x:.2%}")
print("\n📌 최적 포트폴리오 구성 (샤프 비율 최대화):")
print(optimal_portfolio)

# ✅ 종목별 연환산 수익률 출력(콘솔)
print("\n📌 종목별 연환산 수익률 (%)")
print(mean_annual_returns.map(lambda x: f"{x:.2%}"))