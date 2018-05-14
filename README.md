# dynamic-seq2seq
## 欢迎关注我的另一个项目[基于Pytorch以及Beam Search算法的中文聊天机器人](https://github.com/yanwii/seq2seq)
### 基于中文语料和dynamic_rnn的seq2seq模型

---

### Update:
- 修复loss计算bug
- 修复batch_size大于1时的计算bug


### Requirements
- tensorflow-1.4+
- python2.7 (暂时未对python3 兼容)
- requests
- jieba
- cPickle
- numpy

---

谷歌最近开源了一个seq2seq项目 [google seq2seq](https://github.com/google/seq2seq)  
tensorflow推出了dynamic_rnn替代了原来的bucket，本项目就是基于dynamic_rnn的seq2seq模型。  
  
这里我构建了一些对话预料，中文语料本身就比较稀缺，理论上来说语料越多模型的效果越好，但会遇到很多新的问题，这里就不多作说明。   

对话语料分别在**data**目录下 Q.txt A.txt中，可以替换成你自己的对话语料。    

---

### 用法:
    
    # 新增小黄鸡语料
    # 添加
    python prepare_dialog.py 5000


    seq = Seq2seq()
    # 训练
    seq.train()
    # 预测
    seq.predict("天气")
    # 重新训练
    seq.retrain()
   

### 效果:
    
    me >  天气
    AI >  地点： 重庆
    气温： 7
    注意： 天气较凉，较易发生感冒，请适当增加衣服。体质较弱的朋友尤其应该注意防护。

### Action:

本项目添加了Action支持，可以定制自己的功能  
后续会加入多轮会话的支持！  

在action.py文件中,注册自己action标签及对应的接口，如:

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

Tips: 接口的参数暂时固定,后续更新  

同时，训练语料如下设计:

    # Q.txt
    天气
    # A.txt
    __Weather__


