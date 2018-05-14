# -*- coding:utf-8 -*-
import cPickle
import os
import re
import sys
import time

import jieba
import numpy as np
import tensorflow as tf

from action import check_action
from dynamic_seq2seq_model import DynamicSeq2seq
from preprocess import Preprocess
from utils import BatchManager, clear

class Seq2seq():
    '''
    tensorflow-1.0.0

        args:
        encoder_vec_file    encoder向量文件  
        decoder_vec_file    decoder向量文件
        encoder_vocabulary  encoder词典
        decoder_vocabulary  decoder词典
        model_path          模型目录
        batch_size          批处理数
        sample_num          总样本数
        max_batches         最大迭代次数
        show_epoch          保存模型步长

    '''
    def __init__(self):
        print("tensorflow version: ", tf.__version__)
        
        self.dict_file = 'data/word_dict.txt'
        self.data_map = "data/map.pkl"

        self.batch_size = 20
        self.max_epoch = 100000
        self.show_batch = 1
        self.model_path = 'model/'
        # jieba导入词典
        jieba.load_userdict(self.dict_file)

        self.location = ["杭州", "重庆", "上海", "北京"]
        self.user_info = {"__UserName__":"yw", "__Location__":"重庆"}
        self.robot_info = {"__RobotName__":"Rr"}

        # 获取输入输出
        if os.path.isfile(self.data_map):
            with open(self.data_map, "rb") as f: 
                data_map = cPickle.load(f)
        else:
            p = Preprocess()
            p.main()
            data_map = p.data_map

        self.encoder_vocab = data_map.get("Q_vocab")
        self.encoder_vec = data_map.get("Q_vec")
        self.encoder_vocab_size = data_map.get("Q_vocab_size")
        self.char_to_vec = self.encoder_vocab
        
        self.decoder_vocab = data_map.get("A_vocab")
        self.decoder_vec = data_map.get("A_vec")
        self.decoder_vocab_size = data_map.get("A_vocab_size")
        self.vec_to_char = {v:k for k,v in self.decoder_vocab.items()}

        print "encoder_vocab_size {}".format(self.encoder_vocab_size)
        print "decoder_vocab_size {}".format(self.decoder_vocab_size)
        self.model = DynamicSeq2seq(
            encoder_vocab_size=self.encoder_vocab_size+1,
            decoder_vocab_size=self.decoder_vocab_size+1,
        )
        self.sess = tf.Session()
        self.restore_model()
        
    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt:
            print(ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            print("没找到模型")

    def get_fd(self, batch, model):
        '''获取batch

            为向量填充PAD    
            最大长度为每个batch中句子的最大长度  
            并将数据作转换:  
            [batch_size, time_steps] -> [time_steps, batch_size]
        '''
        encoder_inputs = batch[0]
        decoder_targets = batch[1]
        feed_dict = {
            model.encoder_inputs:encoder_inputs,
            model.decoder_targets:decoder_targets
        }
        return feed_dict

    def train(self):
        batch_manager = BatchManager(self.encoder_vec, self.decoder_vec, self.batch_size)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        loss_track = []
        total_time = 0
        nums_batch = len(batch_manager.batch_data)
        for epoch in range(self.max_epoch):
            print "[->] epoch {}".format(epoch)   
            batch_index = 0
            for batch in batch_manager.batch():
                batch_index += 1
                # 获取fd [time_steps, batch_size]
                fd = self.get_fd(batch, self.model)
                _, loss, logits, labels = self.sess.run([self.model.train_op, 
                                    self.model.loss,
                                    self.model.logits,
                                    self.model.decoder_labels], fd)
                loss_track.append(loss)
                if batch_index % self.show_batch == 0:
                    print "\tstep: {}/{}".format(batch_index, nums_batch)
                    print '\tloss: {}'.format(loss)
                    print "\t"+"-"*50
                checkpoint_path = self.model_path+"chatbot_seq2seq.ckpt"
                # 保存模型
                self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        
    def make_inference_fd(self, vec):
        tensor = np.array([vec])
        feed_dict = {
            self.model.encoder_inputs:tensor
        }
        return feed_dict

    def predict(self, input_str):
        # print "Me > ", input_str
        segments = jieba.lcut(input_str)
        vec = [self.char_to_vec.get(seg, 3) for seg in segments]
        feed = self.make_inference_fd(vec)
        logits = self.sess.run([self.model.translations], feed_dict=feed)
        output = logits[0][0].tolist()
        output_str = "".join([self.vec_to_char.get(i, "_UN_") for i in output])
        # check action
        final_output = self.format_output(output_str, input_str)
        # print "AI > ", final_output
        return final_output

    @check_action
    def format_output(self, output_str, raw_input):
        return output_str

    def preprocess(self):
        p = Preprocess()
        p.main()

if __name__ == '__main__':
    if sys.argv[1]:
        if sys.argv[1] == 'retrain':
            clear()
            sys.argv[1] = "train"
        seq = Seq2seq()
        if sys.argv[1] == 'train':
            seq.train()
        elif sys.argv[1] == 'infer':
            print seq.predict("呵呵")  
