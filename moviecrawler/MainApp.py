import collector.DaumCrawler as collector
import requests

code = '127878' # 다음 영화코드

#다음영화 수집 스타트
try:
    scrap = collector.DaumCrawler() # 객체생성
    scrap.crawler(code)
except Exception as e:
    print('>> Exception')
    print('>>', e)
finally:
    pass
