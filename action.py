# -*- coding:utf-8 -*-
import requests

#　注意：参数为固定参数
def act_weather(model, output_str, raw_input):
    #TODO: Get weather by api
    page = requests.get("http://wthrcdn.etouch.cn/weather_mini?city=重庆")
    data = page.json()
    temperature = data['data']['wendu']
    notice = data['data']['ganmao']
    outstrs = "地点： %s\n气温： %s\n注意： %s" % ("重庆", temperature.encode("utf-8"), notice.encode("utf-8"))
    return outstrs

actions = {
    "__Weather__":act_weather
}

def check_action(func):
    def wrapper(*args, **kwargs):
        for i in actions.keys():
            if i in args[1]:
                return actions.get(i)(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper



