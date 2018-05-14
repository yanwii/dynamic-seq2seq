# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-05-14 14:52:35
'''

import re
import sys

def prepare(num_dialogs=1000):
    with open("dialog/xiaohuangji50w_nofenci.conv") as fopen:
        reg = re.compile("E\nM (.*?)\nM (.*?)\n")
        match_dialogs = re.findall(reg, fopen.read())
        if num_dialogs >= len(match_dialogs):
            dialogs = match_dialogs
        else:
            dialogs = match_dialogs[:num_dialogs]
        
        questions = []
        answers = []
        for que, ans in dialogs:
            questions.append(que)
            answers.append(ans)
        save(questions, "dialog/Q")
        save(answers, "dialog/A")

def save(dialogs, file):
    with open(file, "w") as fopen:
        fopen.write("\n".join(dialogs))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_dialogs = int(sys.argv[1])
        prepare(num_dialogs)
    else:
        prepare()