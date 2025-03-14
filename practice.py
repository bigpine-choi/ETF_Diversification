import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

# ✅ 한글 폰트 설정
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

# ✅ 안전자산 & 위험자산 분류
safe_assets = ["153130", "475080", "481340", "434060"]
risky_assets = [ticker for ticker in korean_etfs if ticker not in safe_assets]

safe_indices = [korean_etfs.index(ticker) for ticker in safe_assets]
risky_indices = [korean_etfs.index(ticker) for ticker in risky_assets]

# ✅ 안전자산 & 위험자산 비중 상하한 설정
safe_bounds = (0.3, 1.0)  # 안전자산 최소 30%, 최대 100%
risky_bounds = (0.0, 0.7)  # 위험자산 최소 0%, 최대 70%

# ✅ ETF 종가 데이터 로드 & 수익률 계산
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
df_returns = df_prices.pct_change().dropna()

# ✅ 기대 수익률 & 공분산 계산
trading_days = 252
mean_annual_returns = ((1 + df_returns).prod() ** (trading_days / len(df_returns))) - 1
annual_cov_matrix = df_returns.cov() * trading_days

# ✅ 무위험 수익률 (3%)
risk_free_rate = 0.03

# ✅ 샤프 비율 최적화 함수
def neg_sharpe_ratio(weights):
    portfolio_return = np.sum(weights * mean_annual_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
    return -((portfolio_return - risk_free_rate) / portfolio_volatility)

# ✅ 최적화 실행
bounds = [(0, 1)] * len(korean_etfs)
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: np.sum(x[safe_indices]) - safe_bounds[0]},
    {'type': 'ineq', 'fun': lambda x: safe_bounds[1] - np.sum(x[safe_indices])},
    {'type': 'ineq', 'fun': lambda x: np.sum(x[risky_indices]) - risky_bounds[0]},
    {'type': 'ineq', 'fun': lambda x: risky_bounds[1] - np.sum(x[risky_indices])}
]

initial_weights = np.ones(len(korean_etfs)) / len(korean_etfs)

optimal_result = sco.minimize(
    neg_sharpe_ratio,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

optimal_weights = optimal_result.x

# ✅ 최적 포트폴리오 비중 출력
optimal_portfolio = pd.DataFrame({'ETF': list(etf_names.values()), '비중': optimal_weights})
optimal_portfolio["비중"] = optimal_portfolio["비중"].map(lambda x: f"{x:.2%}")
print("\n📌 최적 포트폴리오 구성:")
print(optimal_portfolio)