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
        sup = 0 if sup == 0 else self.batch_size-sup
        for i in range(sup):
            self.data.append(random.choice(self.data))
        
        index = 0
        while True:
            if index >= len(self.data):
                break

            data = self.data[index:index+self.batch_size]
            paded_data = self.pad(data)
            index += self.batch_size
    
    def pad(self, data):
        Q,A = zip(*data)
        Q_max_len = max([len(i) for i in Q])
        A_max_len = max([len(i) for i in A])

        for vec in Q:
            vec.extend([0] * (Q_max_len-len(vec)))
        
        for vec in A:
            vec.extend([0] * (A_max_len-len(vec)))

        Q = np.array(Q).T
        A = np.array(A).T

        self.batch_data.append([Q,A])

    def batch(self):
        for i in self.batch_data:
            yield i
