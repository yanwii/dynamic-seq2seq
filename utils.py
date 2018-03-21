# -*- coding:utf-8 -*-
import random
import numpy as np
class BatchManager:
    def __init__(self, Q, A, batch_size):
        self.Q = Q
        self.A = A
        self.batch_data = []
        self.batch_size = batch_size
        self.make_batch()

    def make_batch(self):
        assert len(self.Q) == len(self.A), ValueError("问答数据不一致")
        self.data = zip(self.Q, self.A)

        assert len(self.data) > 0, ValueError("训练数据为空")
        sup = len(self.data) % self.batch_size
        sup = 0 if sup == 0 else self.batch_size - sup
        for i in range(sup):
            sup_data = random.choice(self.data)
            self.data.append(sup_data)
        print "-"*50
        index = 0
        while True:
            if index >= len(self.data):
                break
            data = self.data[index:index+self.batch_size]
            padded_data = self.pad(data)
            index += self.batch_size
            self.batch_data.append(padded_data)
    
    def pad(self, data):
        Q,A = zip(*data)
        Q_max_len = max([len(i) for i in Q])
        A_max_len = max([len(i) for i in A])

        new_Q = []
        new_A = []
        for vec in Q:
            new_vec = vec + [0] * (Q_max_len-len(vec))
            new_Q.append(new_vec)
        
        for vec in A:
            new_vec = vec + [0] * (A_max_len-len(vec))
            new_A.append(new_vec)

        Q = np.array(new_Q).T
        A = np.array(new_A).T
        return [Q,A]

    def batch(self):
        for i in self.batch_data:
            yield i
