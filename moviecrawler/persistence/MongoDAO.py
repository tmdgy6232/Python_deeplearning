from pymongo import MongoClient

class MongoDAO:
    def __init__(self):
        # >> MongoDB Connection
        self.client = MongoClient('localhost', 27017)  # 클래스 객체 할당 (ip주소, port번호)
        self.db = self.client['local']
        self.collection = self.db.movie

    def mongo_write(self, data):
        print('>> MongoDB write Data')
        self.collection.insert(data) # Json type = dict type

    def mongo_update(self, data):
        pass

    def mongo_selectAll(self):
        pass

    def mongo_view(self):
        pass

    def mongo_delete(self, data):
        pass
