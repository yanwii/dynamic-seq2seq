import requests

class Action():
    '''
    Action for seq2seq bashed on dynamic rnn      
      
        func:  
        actChangeUsername  
        actWeather  
        actAddPlan  
      
        return:  
        outstrs: None or strs  
        action : True or False 
    '''
    num = 1
    enc_vocab = {}
    dec_vocab = {}
    user_info = {}
    robot_info = {}
    tag_location = ''
    location = []    
    tag_location = ''
    func_id = {}
    
    def __init__(self):
        pass

    def actAddPlan(self, inf_out, inputs_strs):
        outstrs = []
        action = False

        print("ai > 需要建立任务吗？")
        check = input("me > ")
        if check == 'yes' or check == '是的' or check == '好的':
            # add plan
            outstrs.append("已经为您添加计划")

        else:
            outstrs.append("取消添加计划")
        return outstrs, action, inputs_strs
    
    def actWeather(self, inf_out, inputs_strs):
        outstrs = []
        action = False

        location = self.tag_location if self.tag_location else self.user_info['__location__']
        page = requests.get("http://wthrcdn.etouch.cn/weather_mini?city=%s" %(location))
        data = page.json()
        temperature = data['data']['wendu']
        notice = data['data']['ganmao']
        outstrs.append("地点： %s\n     气温： %s\n     注意： %s" % (location, temperature, notice))
        return outstrs, action, inputs_strs

    def actChangeUsername(self, inf_out, inputs_strs):
        outstrs = []
        action = False
        
        print("ai > 您需要我叫您什么呢")
        inputs_strs = input("me > ")
        self.user_info["__username__"] = inputs_strs
        outstrs.append("好的以后就叫您%s了" % (inputs_strs))
        
        return outstrs, action, inputs_strs

    def normalOutputs(self, inf_out, inputs_strs):
        outstrs = []
        action = False
        for vec in inf_out:
            word = self.dec_vocab.get(vec, 3)
            if word == "__EOS__":
                break
            elif word == "__location__":
                outstrs.append(self.tag_location)
            elif "__act" in word:
                continue
            elif word == "__username__":
                outstrs.append(self.user_info.get(word, 3))
            elif word == "__robotname__":
                outstrs.append(self.robot_info["__robotname__"])
            else:
                outstrs.append(self.dec_vocab.get(vec, 3))
        return outstrs, action, inputs_strs

    def main(self, inf_out, inputs_strs):
        func_id = {64:self.actChangeUsername,
                   65:self.actAddPlan,
                   66:self.actWeather}
        func = list(set(inf_out).intersection(set(func_id.keys())))
        #func = None
        if func:
            outstrs, action, inputs_strs = func_id.get(func[0])(inf_out, inputs_strs)
            return outstrs, action, inputs_strs
        else:
            outstrs, action, inputs_strs = self.normalOutputs(inf_out, inputs_strs)
            return outstrs, action, inputs_strs
