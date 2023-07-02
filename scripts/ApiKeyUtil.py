from typing import Union
import requests
from datetime import datetime, timedelta
import json

from pydantic import BaseModel

# 查询使用量URL
usage_url = '/v1/dashboard/billing/usage'
# 查询订阅信息URL
subscription_url = '/v1/dashboard/billing/subscription'

host = 'https://api.openai.com'

class ApiKey(BaseModel):
    key: str
    alive: bool
    amount: float
    used: float


params = {
    # start date is 90 days ago
    'start_date': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
    # end date is tomorrow
    'end_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
}


# set headers and authorization
def get_headers(key: str) -> dict:
    return {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}


def query_key_info(key: str) -> Union[ApiKey, None]:
    usage_response = requests.get(host + usage_url, headers=get_headers(key), params=params)
    subscription_response = requests.get(host + subscription_url, headers=get_headers(key))

    if usage_response.status_code != 200 or subscription_response.status_code != 200:
        return None

    # get response body
    usage_result = usage_response.text
    subscription_result = subscription_response.text

    # 获取总额
    amount = round(json.loads(subscription_result)['hard_limit_usd'], 4)
    # 获取使用量
    used = round(json.loads(usage_result)['total_usage'], 4)
    # divide by 100 to convert from cents to dollars
    used = round(used / 100, 4)

    expire_timestamp = int(json.loads(subscription_result)['access_until'])
    alive = datetime.now().timestamp() < expire_timestamp

    return ApiKey(key=key,
                  alive=alive,
                  amount=amount,
                  used=used)


if __name__ == "__main__":
    apikeyInfo = query_key_info("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")
    print(apikeyInfo)
