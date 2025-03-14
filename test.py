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
safe_assets = ["153130", "475080", "481340", "434060"]
risky_assets = [ticker for ticker in korean_etfs if ticker not in safe_assets]

safe_indices = [korean_etfs.index(ticker) for ticker in safe_assets]
risky_indices = [korean_etfs.index(ticker) for ticker in risky_assets]

# ✅ 안전자산과 위험자산의 비중 상하한 설정
safe_bounds = (0.3, 1.0)  # 안전자산 최소 30%, 최대 100%
risky_bounds = (0.0, 0.7)  # 위험자산 최소 0%, 최대 70%

# ✅ ETF 종가 데이터 로드 및 전처리
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

# ✅ 비중 제약을 적용한 랜덤 포트폴리오 생성
num_portfolios = 50000
results = np.zeros((4, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    safe_weight_total = np.random.uniform(safe_bounds[0], safe_bounds[1])
    risky_weight_total = 1 - safe_weight_total

    safe_weights = np.random.dirichlet(np.ones(len(safe_indices)), size=1)[0] * safe_weight_total
    risky_weights = np.random.dirichlet(np.ones(len(risky_indices)), size=1)[0] * risky_weight_total

    weights = np.zeros(len(korean_etfs))
    weights[safe_indices] = safe_weights
    weights[risky_indices] = risky_weights

    expected_return = np.sum(weights * mean_annual_returns)
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
    sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility

    if not np.isnan(expected_return) and not np.isnan(expected_volatility):
        results[0, i] = expected_return
        results[1, i] = expected_volatility
        results[2, i] = sharpe_ratio
        results[3, i] = i
        weights_record.append(weights)

weights_record = np.array(weights_record)

# ✅ 샤프 비율이 가장 높은 포트폴리오 찾기
max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]

# ✅ 샤프 비율 최적화 함수
def neg_sharpe_ratio(weights):
    portfolio_return = np.sum(weights * mean_annual_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
    return -((portfolio_return - risk_free_rate) / portfolio_volatility)

# ✅ 최적화 실행
bounds = tuple((0, 1) for _ in range(len(korean_etfs)))

optimal_result = sco.minimize(
    neg_sharpe_ratio,
    optimal_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=[
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: np.sum(x[safe_indices]) - safe_bounds[0]},
        {'type': 'ineq', 'fun': lambda x: safe_bounds[1] - np.sum(x[safe_indices])},
        {'type': 'ineq', 'fun': lambda x: np.sum(x[risky_indices]) - risky_bounds[0]},
        {'type': 'ineq', 'fun': lambda x: risky_bounds[1] - np.sum(x[risky_indices])}
    ],
    options={'maxiter': 1000}
)

if optimal_result.success:
    optimal_weights = optimal_result.x

    # ✅ 최적 포트폴리오 성과 재계산
    optimal_return = np.sum(optimal_weights * mean_annual_returns)
    optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(annual_cov_matrix, optimal_weights)))
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

    print("\n📌 최적 포트폴리오 성과:")
    print(f"기대 수익률: {optimal_return:.2%}")
    print(f"변동성: {optimal_volatility:.2%}")
    print(f"샤프 비율: {optimal_sharpe:.2f}")

# ✅ 최적 포트폴리오 비중 출력
optimal_portfolio = pd.DataFrame({'ETF': list(etf_names.values()), '비중': optimal_weights})
optimal_portfolio["비중"] = optimal_portfolio["비중"].map(lambda x: f"{x:.2%}")
print("\n📌 최적 포트폴리오 구성:")
print(optimal_portfolio)

# ✅ 종목별 연환산 수익률 출력
print("\n📌 종목별 연환산 수익률 (%)")
print(mean_annual_returns.map(lambda x: f"{x:.2%}"))