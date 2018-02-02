# -*- coding:utf-8 -*-
import jieba
import re
import os
import cPickle

class Preprocess():
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = {'__PAD__':0, '__GO__':1, '__EOS__':2, '__UNK__':3}
    def __init__(self):
        self.save_dir = "data"
        self.dialog_dir = "dialog"
        self.Q_vocab = self.vocab.copy()
        self.A_vocab = self.vocab.copy()
        self.Q_vec = []
        self.A_vec = []

        self.data_map = {}
 
    def main(self):
        # 加载用户词
        if os.path.isfile(os.path.join(self.save_dir, "word_dict.txt")):
            jieba.load_userdict(os.path.join(self.save_dir, "word_dict.txt"))

        with open(os.path.join(self.dialog_dir, "Q")) as Q_file:
            Qs = [i.strip() for i in Q_file.readlines()]
            self.to_vec("Q", Qs)

        with open(os.path.join(self.dialog_dir, "A")) as A_file:
            As = [i.strip() for i in A_file.readlines()]
            self.to_vec("A", As)

        # save 
        self.data_map = {
            "Q_vocab":self.Q_vocab,
            "Q_vec":self.Q_vec,
            "Q_vocab_size":max(self.Q_vocab.values()),
            "A_vocab":self.A_vocab,
            "A_vec":self.A_vec,
            "A_vocab_size":max(self.A_vocab.values()),
        }
        
        with open(os.path.join(self.save_dir, "map.pkl"),"wb") as f:
            cPickle.dump(self.data_map, f)

    def to_vec(self, dtype, sentences):
        if dtype == "Q":
            vocab = self.Q_vocab
            vec = self.Q_vec
        else:
            vocab = self.A_vocab
            vec = self.A_vec

        max_index = max(vocab.values())
        for sent in sentences:
            segments = jieba.lcut(sent)
            t_vec = []
            for seg in segments:
                if seg not in vocab:
                    vocab[seg] = max_index + 1
                    max_index += 1
                t_vec.append(vocab.get(seg, 3))
            if dtype == "A":
                t_vec.append(2)
            vec.append(t_vec)

        # save vocab 
        with open(os.path.join(self.save_dir, dtype+"_vocab"), "w") as f:
            for k,v in vocab.items():
                f.write("{},{}\n".format(k.encode("utf-8"),v))