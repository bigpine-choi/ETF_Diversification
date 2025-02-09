risk_free_rate = 0.03  # 예시: 한국 3년 국고채 금리 (3.0%)

expected_return = np.sum(weights * mean_returns)
expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility  # 무위험수익률 반영
