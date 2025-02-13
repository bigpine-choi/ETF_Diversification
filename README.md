# ETF_Diversification
국내 ETF 일별 데이터 크롤링 및 퀀트 데이터 분석 &amp; Optimal Portfolio 도출 알고리즘.

✅ 필요한 파일 정리 (ETF 데이터 분석 프로젝트)

1️⃣ Python 패키지 (pip 설치 필요)
각 스크립트에서 사용된 패키지를 설치하려면 다음 명령어 실행:
pip install requests beautifulsoup4 pandas numpy matplotlib seaborn scipy

2️⃣ 각 스크립트 별 필요 파일 목록
파일명	설명
ETF_korean_etf_data_crawler.py	네이버 금융에서 ETF 데이터 크롤링 및 CSV 저장
ETF_korean_etf_data_charts.py	크롤링한 데이터를 시각화 (ETF 가격 변동, 정규화된 수익률 차트)
ETF_korean_etf_data_analysis.py	수익률 분석, 변동성 분석 (30일 이동 표준편차), 상관관계 분석
ETF_korean_etf_data_optimal_portfolio.py	포트폴리오 최적화 (샤프 비율 최대화, 효율적 투자선 계산)

3️⃣ 크롤링된 ETF 데이터 파일 (자동 생성됨)
각 ETF별 CSV 파일이 저장됨 (ETF 코드에 맞는 데이터):
예시 파일명 (실제 실행 시 자동 생성됨):
python-repl
etf_428510_2024-07-16_2025-02-08.csv
etf_456250_2024-07-16_2025-02-08.csv
etf_456600_2024-07-16_2025-02-08.csv
...
⏳ 주의:
ETF_korean_etf_data_crawler.py 실행 후 CSV 파일이 정상적으로 생성되었는지 확인 필요
이후 분석 및 최적화 코드(ETF_korean_etf_data_analysis.py, ETF_korean_etf_data_optimal_portfolio.py)에서 이 파일을 불러옴

4️⃣ 한글 폰트 설정 (Windows / Mac)
각 분석 스크립트에서 한글 폰트를 지정하고 있음.
📌 Windows: Malgun Gothic
📌 Mac: AppleGothic
Mac 사용자는 코드에서 plt.rcParams['font.family'] = 'AppleGothic'로 수정 필요

✅ 실행 순서

1️⃣ ETF 데이터 크롤링:
python ETF_korean_etf_data_crawler.py
🔹 네이버 금융에서 ETF 데이터를 크롤링하여 CSV 파일 저장

2️⃣ ETF 데이터 시각화:
python ETF_korean_etf_data_charts.py
🔹 ETF 가격 변동 및 정규화된 수익률 비교 그래프 생성

3️⃣ ETF 데이터 분석:
python ETF_korean_etf_data_analysis.py
🔹 수익률 변동, 변동성 분석(30일 이동 표준편차), 상관관계 분석 수행

4️⃣ 포트폴리오 최적화:
python ETF_korean_etf_data_optimal_portfolio.py
🔹 샤프 비율 최대화 포트폴리오 최적화 및 효율적 투자선 시각화
