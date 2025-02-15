from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Selenium용 Chrome 드라이버 설정
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# 대상 URL
url = 'https://m.samsungfund.com/etf/product/view.do?id=2ETFM1'
driver.get(url)

# 페이지 로딩 대기 (필요에 따라 조정)
time.sleep(3)

# 분배금 지급 현황 테이블 찾기
# (태그와 클래스를 실제 페이지 구조에 맞게 수정해야 합니다)
try:
    dividend_table = driver.find_element(By.CLASS_NAME, 'tbl_type')
    rows = dividend_table.find_elements(By.TAG_NAME, 'tr')

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, 'td')
        cols = [col.text.strip() for col in cols]
        print(cols)
except Exception as e:
    print("분배금 지급 현황 테이블을 찾을 수 없습니다.", e)

# 브라우저 종료
driver.quit()