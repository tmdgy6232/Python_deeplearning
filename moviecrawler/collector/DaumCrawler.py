from bs4 import BeautifulSoup
from selenium import webdriver
import persistence.MongoDAO as DAO
import requests


class DaumCrawler():

    def __init__(self):
        self.mDao = DAO.MongoDAO() # 객체생성

    def crawler(self, code):

        #페이지 존재유뮤 체크
        doc = requests.get('https://movie.daum.net/moviedb/grade?movieId={}'.format(code))
        if doc.status_code != 200: # 200 sucess
            print('>> Not Found Page:/')
            return
        count = 0  # 댓글 총 갯수
        page = 0  # page 넘버

        while True:
            page += 1
            url = 'https://movie.daum.net/moviedb/grade?movieId={}&type=netizen&page={}'.format(code, page)

            path = 'E:\Bigdata\webdriver\chromedriver_win32\chromedriver.exe'
            driver = webdriver.Chrome(path)
            driver.get(url)  # http://까지 적어야함
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            reply_list = soup.select('ul.list_review > li')
            if reply_list == []:  # 다음 페이지에 아무것도 뜨지 않으면, 브레이크해라
                page = page - 1
                break
            else:
                pass
            for reply in reply_list:
                content = reply.select('p.desc_review')[0].text.strip()
                writer = reply.select('div > a > em.link_profile')[0].text.strip()
                score = reply.select('div em.emph_grade')[0].text.strip()
                reg_date = reply.select('div span.info_append')[0].text.strip()
                indexNo = reg_date.find(',')
                reg_date = reg_date[:indexNo]

                # mongoDB에 댓글 저장
                data = {
                    'score': score,
                    'writer': writer,
                    'reg_date': reg_date,
                    'content': content}

                self.mDao.mongo_write(data)

                print('▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒')
                print('평점 : ', score)
                print('작성자 : ', writer)
                print('작성일자 : ', reg_date)
                print('내용 : ', content)
                count += 1

        driver.close() #드라이버 자원 반납
        print('▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒')
        print('총 ', page, '페이지, 데이터는 ', count, '건 출력되었습니다.')
