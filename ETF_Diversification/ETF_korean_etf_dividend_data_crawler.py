import requests
from bs4 import BeautifulSoup

# 대상 URL
url = 'https://m.samsungfund.com/etf/product/view.do?id=2ETFM1'

# HTTP GET 요청
response = requests.get(url)
response.raise_for_status()  # 요청에 실패할 경우 예외 발생

# HTML 파싱
soup = BeautifulSoup(response.text, 'html.parser')

# 분배금 지급 현황 테이블 찾기
# (태그와 클래스를 실제 페이지 구조에 맞게 수정해야 합니다)
dividend_table = soup.find('table', {'class': 'tbl_type'})

# 테이블이 존재하는지 확인
if dividend_table:
    # 테이블의 모든 행(tr) 찾기
    rows = dividend_table.find_all('tr')

    for row in rows:
        # 각 행의 모든 열(td) 찾기
        cols = row.find_all('td')
        # 열의 텍스트 추출
        cols = [col.get_text(strip=True) for col in cols]
        print(cols)
else:
    print("분배금 지급 현황 테이블을 찾을 수 없습니다.")