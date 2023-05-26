import os
from datetime import datetime
from peewee import *

import logging

logger = logging.getLogger('peewee')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

mysql_db = MySQLDatabase(os.getenv("MYSQL_DATABASE"), host=os.getenv("MYSQL_HOST"), user=os.getenv("MYSQL_USER"), password=os.getenv("MYSQL_PASSWORD"))


class BaseModel(Model):
    class Meta:
        database = mysql_db


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


class UserToken(BaseModel):
    id = AutoField()
    user_id = CharField(unique=True)
    tokens_purchased = FloatField(default=0)
    tokens_consumed = FloatField(default=0)

    class Meta:
        table_name = 'user_tokens'


class ApiKey(BaseModel):
    id = AutoField()
    apikey = CharField(unique=True)
    is_alive = BooleanField()
    total_amount = DoubleField()
    consumption = DoubleField()

    class Meta:
        table_name = 'apikeys'


class ChatPaper(BaseModel):
    id = AutoField()
    pdf_hash = CharField()
    query = TextField()
    user_id = CharField()
    content = TextField()
    token_cost = IntegerField()
    pages = CharField()
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'ChatPaper'


class PaperDocVector(BaseModel):
    id = AutoField()
    pdf_hash = CharField(null=True)
    page_content = TextField()
    vector_id = IntegerField()
