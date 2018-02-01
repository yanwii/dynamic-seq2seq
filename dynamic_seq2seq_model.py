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
        time_major              控制输入数据格式
                                True:  [time_steps, batch_size]
                                False: [batch_size, time_steps]

        
    '''
    PAD = 0
    EOS = 2
    UNK = 3
    def __init__(self, 
                encoder_cell=tf.contrib.rnn.BasicLSTMCell(10), 
                decoder_cell=tf.contrib.rnn.BasicLSTMCell(10), 
                encoder_vocab_size=10,
                decoder_vocab_size=5, 
                embedding_size=10,
                bidirectional=True,
                attention=False,
                debug=False,
                time_major=False):
        
        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention
        self.lstm_dims = 10

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        
        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell
        
        self.global_step = tf.Variable(-1, trainable=False)
        self.max_gradient_norm = 5
        self.time_major = time_major

        #创建模型
        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size
    
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
        #self._init_optimizer()

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
        )

        self.batch_size = tf.shape(self.encoder_inputs)[1]

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
            self.decoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_decoder, self.decoder_targets
                )
            
    def _init_bidirectional_encoder(self):
        '''
        双向LSTM encoder
        '''
        # Build RNN cell
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dims)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, self.encoder_emb_inp,
            sequence_length=self.encoder_inputs_length, time_major=True,
            dtype=tf.float32
        )
        self.encoder_output = encoder_outputs
        self.encoder_state = encoder_state

    def _init_decoder(self):
        attention_states = tf.transpose(self.encoder_output, [1, 0, 2])

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.lstm_dims, 
            memory=attention_states,
            memory_sequence_length=self.decoder_targets_length
        )

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell, attention_mechanism,
            attention_layer_size=self.lstm_dims
        )
        # Helper    
        helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_emb_inp, 
            self.decoder_targets_length, 
            time_major=True
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
        maximum_iterations = tf.round(tf.reduce_max(self.decoder_targets_length) * 2)
        # Dynamic decoding
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder, 
            maximum_iterations=maximum_iterations
        )
        self.logits = outputs

    def _init_optimizer(self):
        # 整理输出并计算loss
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        self.targets = tf.transpose(self.decoder_train_targets, [1, 0])
    
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)
        
        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss)

        # add
        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []

        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def run(self):
        feed = {
            self.encoder_inputs:[[2],[1],[2],[3],[4]],
            self.encoder_inputs_length:[5],
            self.decoder_targets:[[1],[0],[4],[3],[2]],
            self.decoder_targets_length:[5]
        }

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            logits = sess.run([self.logits], feed_dict=feed)
            print logits[0][0]

model = DynamicSeq2seq()
model.run()