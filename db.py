import os
import datetime
from peewee import *
from dotenv import load_dotenv

if os.getenv('ENV') == 'DEV':
    load_dotenv()

import logging

logger = logging.getLogger('peewee')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

mysql_db = MySQLDatabase(os.getenv("MYSQL_DATABASE"), host=os.getenv("MYSQL_HOST"), user=os.getenv("MYSQL_USER"), password=os.getenv("MYSQL_PASSWORD"))

mysql_db_new = MySQLDatabase(os.getenv("MYSQL_DATABASE_NEW"), host=os.getenv("MYSQL_HOST"),
                             user=os.getenv("MYSQL_USER"), password=os.getenv("MYSQL_PASSWORD"))

class BaseModel(Model):
    class Meta:
        database = mysql_db

class BaseModelNew(Model):
    class Meta:
        database = mysql_db_new


class User(BaseModel):
    id = AutoField()
    user_id = CharField(unique=True)
    email = CharField()
    user_name = CharField()
    token_consumed = FloatField(default=0)
    vip_level = IntegerField()
    password = CharField(null=True)
    disabled = BooleanField(default=False)
    is_Google = BooleanField(default=False)

    class Meta:
        table_name = 'usersNew'

class Accounts(BaseModelNew):
    id = AutoField()
    user_id = CharField()
    provider = CharField()
    provider_account_id = CharField()
    refresh_token = CharField()
    access_token = TextField()
    expires_at = IntegerField()
    token_type = CharField()
    scope = CharField()
    id_token = TextField()
    session_state = CharField()
    type = CharField(choices=("oauth", "email", "credentials"))
    class Meta:
        table_name = 'accounts'

class Users(BaseModelNew):
    id = AutoField()
    name = CharField()
    email = CharField()
    email_verified = DateTimeField()
    image = CharField()
    password = CharField()
    credits = FloatField()
    class Meta:
        table_name = 'users'

class CreditHistories(BaseModelNew):
    id = AutoField()
    user_id = CharField()
    amount = FloatField()
    type = CharField(choices=('SIGN', 'PURCHASE', 'TASK'))
    happenedAt = DateTimeField(default=datetime.datetime.now)
    class Meta:
        table_name = 'credit_histories'


class UserToken(BaseModel):
    id = AutoField()
    user_id = CharField(unique=True)
    tokens_purchased = FloatField(default=0)
    tokens_consumed = FloatField(default=0)

    class Meta:
        table_name = 'user_tokens'


class ApiKey(BaseModelNew):
    id = AutoField()
    key = CharField(unique=True)
    alive = BooleanField()
    amount = DoubleField()
    used = DoubleField()

    class Meta:
        table_name = 'api_keys'


class ChatPaper(BaseModel):
    id = AutoField()
    pdf_hash = CharField()
    query = TextField()
    user_id = CharField()
    content = TextField()
    token_cost = IntegerField()
    pages = CharField()
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = 'ChatPaper'


class PaperDocVector(BaseModel):
    id = AutoField()
    pdf_hash = CharField(null=True)
    page_content = TextField()
    vector_id = IntegerField()



class PaperInfo(BaseModelNew):
    id = AutoField(primary_key=True)
    url = CharField()   # 文章网页连接
    pdf_url = CharField()   # pdf url
    eprint_url = CharField()   # 预印版pdf url
    pdf_hash = CharField()  # pdf hash
    year = IntegerField()   # 年份
    title = CharField()
    venue = CharField()
    conference = CharField()
    url_add_scib = CharField()
    bibtex = TextField()
    url_scholarbib = CharField()
    code = CharField()
    num_citations = IntegerField()
    cited_by_url = CharField()
    url_related_articles = CharField()  # 相关文章链接
    authors = CharField()
    abstract = TextField()
    img_url = CharField()
    pub_time = DateTimeField()
    keywords = CharField()
    create_time = DateTimeField(default=datetime.datetime.now)
    doi = CharField()

    class Meta:
        table_name = 'paper_info'

class UserTasks(BaseModelNew):
    id = CharField(primary_key=True)
    user_id = CharField()
    pdf_hash = CharField()
    language = CharField()
    type = CharField(choices=('SUMMARY', 'TRANSLATE'))
    state = CharField(choices=('WAIT', 'RUNNING', 'SUCCESS', 'FAIL'))
    created_at = DateTimeField(default=datetime.datetime.now)
    finished_at = DateTimeField(null=True)
    cost_credits = IntegerField()   #
    pages = IntegerField()

    class Meta:
        table_name = 'tasks'


# 任务表
class SubscribeTasks(BaseModelNew):
    id = AutoField(primary_key=True)
    type = CharField(choices=('SUMMARY', 'TRANSLATE'))
    tokens = IntegerField()
    pages = IntegerField()
    pdf_hash = CharField()
    language = CharField()
    state = CharField(choices=('WAIT', 'RUNNING', 'SUCCESS', 'FAIL'))
    created_at = DateTimeField(default=datetime.datetime.now)
    finished_at = DateTimeField()
    class Meta:
        table_name = 'subscribe_tasks'

# 总结表
class Summaries(BaseModelNew):
    id = AutoField(primary_key=True)
    pdf_hash = CharField()
    language = CharField()
    title = CharField()
    title_zh = CharField()
    basic_info = CharField()
    brief_introduction = CharField()
    first_page_conclusion = CharField()
    content = CharField()
    medium_content = CharField()
    short_content = CharField()
    create_time = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = 'summaries'


def test():
    # query_api_keys = ApiKey.select().order_by(ApiKey.consumption.asc()).where(
    #     ApiKey.is_alive == True)
    # keys = [apikey.apikey for apikey in query_api_keys]
    # print(keys)
    #
    # query_search_keywords = SearchKeys.select()
    # search_keywords = [keywords.search_keywords for keywords in query_search_keywords]
    # keyword_short = [keywords.keyword_short for keywords in query_search_keywords]
    # print(search_keywords)

    # test tasks
    # 创建任务对象
    task = SubscribeTasks(id='1234',
                          type='SUMMARY',
                          tokens=0,
                          state='RUNNING',
                          pdf_hash='file_hash',
                          pages=0,
                          language='中文',
                          created_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # 保存任务到数据库
    task.save()
    print(task.id)


if __name__ == "__main__":
    test()
