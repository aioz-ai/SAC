import numpy as np


import tensorflow as tf

np.random.seed(1)
rnn_cell = tf.contrib.rnn


def class_embedding(sentence, word2vec, word2index, emb_dim):
    batch, num_class, max_words = sentence.shape
    rnn_size = 1024
    sentence = tf.reshape(sentence, [batch*num_class, max_words])
    sentence = tf.cast(sentence, dtype=tf.int32)
    # create word embedding
    embed_ques_W = tf.Variable(word2vec)
    # create LSTM
    lstm_1 = rnn_cell.LSTMCell(rnn_size, emb_dim)
    lstm_dropout_1 = rnn_cell.DropoutWrapper(lstm_1, output_keep_prob=0.8)
    lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size)
    lstm_dropout_2 = rnn_cell.DropoutWrapper(lstm_2, output_keep_prob=0.8)
    stacked_lstm = rnn_cell.MultiRNNCell([lstm_dropout_1, lstm_dropout_2])
    state = stacked_lstm.zero_state(batch*num_class, tf.float32)

    with tf.variable_scope("embed"):
        for i in range(max_words):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            cls_emb_linear = tf.nn.embedding_lookup(embed_ques_W, sentence[:, i])
            cls_emb_drop = tf.nn.dropout(cls_emb_linear, .8)
            cls_emb = tf.tanh(cls_emb_drop)

            output, state = stacked_lstm(cls_emb, state)
    output = tf.reshape(output, [batch, num_class, rnn_size])
    return output