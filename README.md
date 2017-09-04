# dynamic-seq2seq
## 欢迎关注我的另一个项目[基于Pytorch以及Beam Search算法的中文聊天机器人](https://github.com/yanwii/seq2seq)
### 基于中文语料和dynamic_rnn的seq2seq模型

**需要 python3+ tensorflow-1.0**  
**由于tensorflow升级 本教程只适合tesorflow-1.0版本**  

谷歌最近开源了一个seq2seq项目 [google seq2seq](https://github.com/google/seq2seq)  
这个项目加入了beam search，但是非官方的项目，并且该项目是直接从文件里面读数据，所以需要修改代码。  
tensorflow推出了dynamic_rnn替代了原来的bucket，本项目就是基于dynamic_rnn的seq2seq模型。  
  
这里我构建了一些对话预料，中文语料本身就比较稀缺，理论上来说语料越多模型的效果越好，但会遇到很多新的问题，这里就不多作说明。   
~~我在模型中加入了**Action**，可以实现简单的功能，算是Demo吧。~~
删除Action 交给你们自己去实现吧


对话语料分别在根目录下 question.txt answer.txt中，可以替换成你自己的对话语料。    
然后使用preprocessing.py自动化预处理。
### 用法:
    # 预处理
    python3 preprocessing.py
    # 训练
    python3 seq2seq.py train
    # 重新训练
    python3 seq2seq.py retrain
    # 预测
    python3 seq2seq.py infer
   
  
### 效果:
    
    me > 你的名字
    RR > 我叫RR
    
    me > 你
    RR > 我是RR呀，请问有什么可以帮您吗？
    
    me > 天气
    RR > 地点： 重庆
         气温： 27
         注意： 各项气象条件适宜，无明显降温过程，发生感冒机率较低。

    me > 北京的天气
    RR > 地点： 北京
         气温： 26
         注意： 各项气象条件适宜，无明显降温过程，发生感冒机率较低。
         
    me > 我是谁
    RR > 您是yw
    me > 修改我的名字
    ai > 您需要我叫您什么？
    me > 程序猿
    RR > 好的以后就叫您程序猿了
    me > 我的名字
    RR > 您是程序猿
