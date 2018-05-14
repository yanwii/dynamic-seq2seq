# -*- coding:utf-8 -*-
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell

class DynamicSeq2seq():
    '''
    Dynamic_Rnn_Seq2seq with Tensorflow-1.0.0  

        args:
        encoder_cell            encoder结构
        decoder_cell            decoder结构
        encoder_vocab_size      encoder词典大小
        decoder_vocab_size      decoder词典大小
        embedding_size          embedd成的维度
        bidirectional           encoder的结构
                                True:  encoder为双向LSTM 
                                False: encoder为一般LSTM
        attention               decoder的结构
                                True:  使用attention模型
                                False: 一般seq2seq模型
                      控制输入数据格式
                                True:  [time_steps, batch_size]
                                False: [batch_size, time_steps]

        
    '''
    PAD = 0
    EOS = 2
    UNK = 3
    def __init__(self, 
                encoder_cell=tf.contrib.rnn.BasicLSTMCell(40), 
                decoder_cell=tf.contrib.rnn.BasicLSTMCell(40), 
                encoder_vocab_size=10,
                decoder_vocab_size=5, 
                embedding_size=10,
                attention=False,
                debug=False,
                time_major=False):
        
        self.debug = debug
        self.attention = attention
        self.lstm_dims = 40

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        
        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell
        
        self.global_step = tf.Variable(-1, trainable=False)
        self.max_gradient_norm = 5

        #创建模型
        self._make_graph()

    def _make_graph(self):
        # 创建占位符
        self._init_placeholders()

        # embedding层
        self._init_embeddings()

        # 判断是否为双向LSTM并创建encoder
        self._init_bidirectional_encoder()

        # 创建decoder，会判断是否使用attention模型
        self._init_decoder()

        # 计算loss及优化
        self._init_optimizer()

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        self.decoder_inputs = tf.concat([tf.ones(shape=[self.batch_size, 1], dtype=tf.int32), self.decoder_targets], 1)
        self.decoder_labels = tf.concat([self.decoder_targets, tf.zeros(shape=[self.batch_size, 1], dtype=tf.int32)], 1)

        used = tf.sign(tf.abs(self.encoder_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.encoder_inputs_length = tf.cast(length, tf.int32)

        used = tf.sign(tf.abs(self.decoder_labels))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.decoder_targets_length = tf.cast(length, tf.int32)


    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:

            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            # encoder Embedding
            embedding_encoder = tf.get_variable(
                    "embedding_encoder", 
                    shape=[self.encoder_vocab_size, self.embedding_size],
                    initializer=initializer,
                    dtype=tf.float32
                )
            self.encoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_encoder, self.encoder_inputs
                )
            #  decoder Embedding
            embedding_decoder = tf.get_variable(
                    "embedding_decoder", 
                    shape=[self.decoder_vocab_size, self.embedding_size],
                    initializer=initializer,
                    dtype=tf.float32
                )
            self.embedding_decoder = embedding_decoder
            self.decoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_decoder, self.decoder_inputs
                )
            
    def _init_bidirectional_encoder(self):
        '''
        双向LSTM encoder
        '''
        # Build RNN cell
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dims)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, self.encoder_emb_inp,
            sequence_length=self.encoder_inputs_length, time_major=False,
            dtype=tf.float32
        )
        self.encoder_output = encoder_outputs
        self.encoder_state = encoder_state

    def _init_decoder(self):
        # attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
        attention_states = self.encoder_output

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.lstm_dims, 
            memory=attention_states,
        )

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell, attention_mechanism,
            attention_layer_size=self.lstm_dims
        )
        # Helper    
        helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_emb_inp, 
            self.decoder_targets_length+1, 
            time_major=False
        )
        projection_layer = tf.layers.Dense(self.decoder_vocab_size, use_bias=False)
        init_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=helper,
            initial_state=init_state,
            output_layer=projection_layer
        )
        maximum_iterations = tf.round(tf.reduce_max(self.encoder_inputs_length) * 20)
        # Dynamic decoding
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder, 
            maximum_iterations=maximum_iterations
        )
        self.logits = outputs

        # ------------Infer-----------------
        # Helper
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_decoder,
            tf.fill([self.batch_size], 1), 2)
    
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=infer_helper,
            initial_state=init_state,
            output_layer=projection_layer
            )
        # Dynamic decoding
        infer_outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_iterations)
        self.translations = infer_outputs[0][1]
        

    def _init_optimizer(self):
        # 整理输出并计算loss
        mask = tf.sequence_mask(
            tf.to_float(self.decoder_targets_length),
            tf.to_float(tf.shape(self.decoder_labels)[1])    
        )
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.logits[0][0],
            self.decoder_labels,
            tf.to_float(mask)
        )

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                        gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        self.train_op = update_step
        self.saver = tf.train.Saver(tf.global_variables())

    def run(self):
        feed = {
            self.encoder_inputs:[[2,1],[1,2],[2,3],[3,4],[4,5]],
            self.decoder_targets:[[1,1],[1,1],[4,1],[3,1],[2,0]],
        }
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                logits,_,loss = sess.run([self.logits, self.train_op, self.loss], feed_dict=feed)