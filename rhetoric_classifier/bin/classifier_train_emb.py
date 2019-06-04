'''
   Created by zhiqliu on 2018/12/18.
'''

import os
import subprocess
import time
import codecs
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import reader
import conf
from log import g_log_inst
from sklearn import metrics
np.set_printoptions(threshold=np.nan)

# global variables
flags = tf.flags
logging = tf.logging

# setting input parser
flags.DEFINE_string('log_dir', '../log/summary', 'write log dir for showing of tensorboard') # write log dir for showing of tensorboard
flags.DEFINE_string('model', 'bilstm_attention_mdl','A type of model. Possible options are: bilstm_attention_mdl.') # model options
flags.DEFINE_string('data_path', '../data2', 'directory where stores training datasets and vocab') # data file path
FLAGS = flags.FLAGS


class BiLSTM_Attention_Model(object):
    # init model
    def __init__(self, is_training, config):
        # config
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.num_labels = num_labels = config.num_labels
        self.attention_size = attention_size = config.attention_size

        hidden_size = config.hidden_size
        num_layers = config.num_layers
        keep_prob = config.keep_prob
        max_grad_norm = config.max_grad_norm
        vocab_size = config.vocab_size

        # feed
        self._input_data = tf.placeholder(tf.int32, [None, num_steps], name='input_x')
        self._targets = tf.placeholder(tf.int32, [None, 1], name='labels')
        self._sequence_length = tf.placeholder(tf.int32, [batch_size])
        
        # embedding
        with tf.device('/cpu:0'):
            #embedding = tf.get_variable('embedding', [vocab_size, hidden_size], dtype=tf.float32)
            #inputs = tf.nn.embedding_lookup(embedding, self._input_data)
            emb_matrix = np.load("../data2/word2vec_new.model.npy")
            embedding = tf.get_variable("embedding", shape=emb_matrix.shape,
                                        initializer=tf.constant_initializer(emb_matrix), trainable=False)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)            
        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)
        
        cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)

        if is_training and keep_prob < 1:
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=config.keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=config.keep_prob)

        cells_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers, state_is_tuple=True)
        cells_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers, state_is_tuple=True)
        
        self._initial_state_fw = initial_state_fw = cells_fw.zero_state(batch_size, tf.float32)
        self._initial_state_bw = initial_state_bw = cells_bw.zero_state(batch_size, tf.float32)
        
        x = tf.transpose(inputs, [1, 0, 2])
        # # Reshape to (num_steps * batch_size, hidden_size)
        x = tf.reshape(x, [-1, hidden_size])

        x = tf.split(axis=0, num_or_size_splits=num_steps, value=x)
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cells_fw, cells_bw, x,
                                                     initial_state_fw=initial_state_fw,
                                                     initial_state_bw=initial_state_bw, 
													 dtype=tf.float32)


        # Attention layer
        output, self.alphas = self.attention(outputs, attention_size)
        # output size of softmax is the label numbers
        with tf.name_scope('softmax_weight'):
            softmax_w = tf.get_variable('softmax_w', [hidden_size * 2, num_labels], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [num_labels], dtype=tf.float32)
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(output , softmax_w) + softmax_b
                tf.summary.histogram('logits', logits)
                result = tf.nn.softmax(logits=logits)
                print (result)
            self._predicts = tf.argmax(logits, 1)
            self._logits = result[0]
            print (self._logits)
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.to_int32(tf.argmax(logits, 1), name='ToInt32'), tf.reshape(self._targets, [-1]))
                self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.summary.scalar('accuracy', self._accuracy)
        # use sparse Softmax because we have mutually exclusive classes
        with tf.name_scope('train_loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self._targets, [-1]), name='Sparse_softmax')
            self._cost = cost = tf.reduce_sum(loss) / batch_size
        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)
        tf.add_to_collection('input_x',self._input_data)
        tf.add_to_collection('labels',self._targets)
        tf.add_to_collection('score',self._logits)
        #tf.add_to_collection('targets',self._targets)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def merged(self):
        return tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    @property
    def targets(self):
        return self._targets

    @property
    def sequence_length(self):
        return self._sequence_length
    
    @property
    def initial_state(self):
        return self._initial_state_fw

    @property
    def cost(self):
        return self._cost

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def predicts(self):
        return self._predicts

    @property
    def logits(self):
        return self._logits

    def attention(self, inputs, attention_size, time_major=False):

        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        if time_major:
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        inputs = tf.transpose(tf.stack(inputs), perm=[1, 0, 2])
        sequence_length = inputs.shape[1].value
        hidden_size = inputs.shape[2].value  

        # Attention mechanism
        with tf.name_scope('attention_weight'):
            W_omega = tf.get_variable('W_omega', [hidden_size, attention_size], dtype=tf.float32)
            b_omega = tf.get_variable('b_omega', [attention_size], dtype=tf.float32)
            u_omega = tf.get_variable('u_omega', [attention_size], dtype=tf.float32)
            with tf.name_scope('attention_wx_plus_b'):
                v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
                vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
                exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
                alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
            tf.summary.histogram('v', v)
            tf.summary.histogram('vu', vu)
            tf.summary.histogram('exps', exps)
            tf.summary.histogram('alphas', alphas)

        # Output of Bi-LSTM is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        return output, alphas


def run_epoch(session, model, data, eval_op, debug=False, verbose=False, id2word_dict=None, dsl_converter=None):
    '''Runs the model on the given data.'''
    epoch_size = len(data) // model.batch_size
    # statistic
    start_time = time.time()
    costs = 0.0
    iters = 0
    accuracy_sum = 0.0
    domain_accuracy = [[0, 0, 0, 0] for i in range(
        model.num_labels)]  # [pred1, pre2d, rec1, rec2], calculate the precision and recall at the same time
    
    state = session.run(model.initial_state)
    if debug:
        wrong_predict_out = codecs.open('../log/wrong_pred_emb.txt', 'w', 'utf-8')
    y_label = []
    y_pred = []
    for step, (data_ids, sequences, labels, seq_lens) in enumerate(
            reader.pairs_iterator(data, model.batch_size, model.num_steps)):
        #print sequences
        feed_dict = {}
        feed_dict[model.input_data] = sequences
        feed_dict[model.targets] = labels
        feed_dict[model.sequence_length] = seq_lens
        
        if debug:
            fetches = [model.cost, model.predicts, eval_op]
            cost, predicts, _ = session.run(fetches, feed_dict)
            x_str = '_'.join(map(lambda x: str(x), sequences[0]))
            domain_predict = predicts[-1]
            domain_label = labels[0][0]
            y_label.append(dsl_converter.label2domain[int(domain_label)])
            y_pred.append(dsl_converter.label2domain[int(domain_predict)])
            domain_accuracy[domain_predict][1] += 1  # denominator of precision
            domain_accuracy[int(domain_label)][3] += 1  # denominator of recall
            accuracy = 1 if str(domain_predict) == str(domain_label) else 0
            # if right, update the pred and rec
            if accuracy == 1:
                domain_accuracy[int(domain_label)][0] += 1  # nominator of precision
                domain_accuracy[domain_predict][2] += 1  # nominator of recall
            else:
                raw_str = '|'.join([str(i) for i in sequences[0][0:seq_lens[0]]])
                #wrong_predict_out.write('data_id=%s, label=%s, predict=%s, raw_str=%s\n' % (
                #data_ids, dsl_converter.label2domain[int(domain_label)], dsl_converter.label2domain[domain_predict],
                #raw_str))
                wrong_predict_out.write('%s\t%s\t%s\n' %(str(data_ids).replace("'",'').replace('[','').replace(']',''), str(domain_label), str(domain_predict)))
            g_log_inst.get().debug('step=%s, cost=%s, x_str=%s, predict=%s, label_idx=%s' % (
            step, cost, x_str, domain_predict, domain_label))
            g_log_inst.get().debug('predicts=%s' % (predicts))

        else:
            fetches = [model.cost, model.accuracy, eval_op]
            cost, accuracy, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.num_steps

        accuracy_sum += accuracy
        avg_accuracy = accuracy_sum / (step + 1)

        if verbose and step % (epoch_size // 10) == 10:
            g_log_inst.get().info('%.3f perplexity: %.3f speed: %.0f wps, accuracy=%.03f' % (
            step * 1.0 / epoch_size, np.exp(costs / iters), iters * model.batch_size / (time.time() - start_time),
            avg_accuracy))

    if debug:
        y_true = y_label
        classify_report = metrics.classification_report(y_true, y_pred)
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        overall_accuracy = metrics.accuracy_score(y_true, y_pred)
        acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
        average_accuracy = np.mean(acc_for_each_class)
        score = metrics.accuracy_score(y_true, y_pred)
        print('classify_report : \n', classify_report)
        print('average_accuracy: {0:f}'.format(average_accuracy))
        print('overall_accuracy: {0:f}'.format(overall_accuracy))
        print('score: {0:f}'.format(score))
    
    if debug:
        wrong_predict_out.close()
    perplexity = np.exp(costs / iters)
    return (perplexity, avg_accuracy, domain_accuracy)

def get_config():
    if FLAGS.model == 'bilstm_attention_mdl':
        return conf.DefaultConfig()
    else:
        raise ValueError('Invalid model: %s', FLAGS.model)

def tokenizer_fn(iterator):
    return (x.split('|') for x in iterator)

def main(_):
    if not FLAGS.data_path:
        g_log_inst.get().error('Must set --data_path to training data directory')
        return

    ckpt_dir = '../model_emb/training-ckpt/' + FLAGS.model
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1

    [train_data, valid_data, test_data], id2word_dict = reader.load_train_data(FLAGS.data_path)
    g_log_inst.get().info('bilstm-attention training begin')
    
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.model, reuse=None, initializer=initializer):
            m = BiLSTM_Attention_Model(is_training=True, config=config)
        with tf.variable_scope(FLAGS.model, reuse=True, initializer=initializer):
            mtest = BiLSTM_Attention_Model(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        # add ops to save and restore all the variables.
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            g_log_inst.get().info('[model] restore success, ckpt_path=%s' % (ckpt.model_checkpoint_path))
            save_path = saver.save(session, '%s/model.ckpt' % (ckpt_dir))
        else:
            pre_valid_perplexity = float("inf")
            learning_rate = config.learning_rate
            start_decay = False
            for i in range(config.max_epoch):
                if start_decay == True:
                    learning_rate *= config.lr_decay
                m.assign_lr(session, learning_rate)
                g_log_inst.get().info('Epoch: %d Learning rate: %.3f' % (i + 1, session.run(m.lr)))
                # shuffle the data before mini-batch training
                random.shuffle(train_data)

                # train
                train_perplexity, accuracy, _  = run_epoch(session, m, train_data, m.train_op, verbose=True)
                g_log_inst.get().info(
                    'Epoch: %d Train Perplexity: %.3f accuracy: %s' % (i + 1, train_perplexity, accuracy))

                # valid
                valid_perplexity, valid_accuracy, _ = run_epoch(session, mtest, valid_data, tf.no_op())

                g_log_inst.get().info(
                    'Epoch: %d Valid Perplexity: %.3f accuracy: %s' % (i + 1, valid_perplexity, valid_accuracy))

                # if valid set perplexity improves too small, start lr decay
                if pre_valid_perplexity - valid_perplexity < config.perplexity_thres and  start_decay:
                    start_decay = False
                    g_log_inst.get().info('Valid Perplexity increases too small, start lr decay')
                if pre_valid_perplexity < valid_perplexity:  # current epoch is rejected
                    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(session, ckpt.model_checkpoint_path)
                        g_log_inst.get().info('[model] restore success, ckpt_path=%s' % (ckpt.model_checkpoint_path))
                    if learning_rate == config.learning_rate:  # if has not decay yet, give a second chance
                        continue
                    else:  # stop training
                        g_log_inst.get().info('Valid Perplexity does not increase, stop training')
                        break

                pre_valid_perplexity = valid_perplexity
                # save the variables to disk.
                save_path = saver.save(session, '%s/model.ckpt' % (ckpt_dir))
                g_log_inst.get().info('[model] save success, ckpt_path=%s' % (save_path))

        # test the accuracy
        test_perplexity, accuracy, domain_accuracy = run_epoch(session, mtest, test_data, tf.no_op(), debug=True,
                                                               verbose=True, id2word_dict=id2word_dict, dsl_converter=config.converter)
        g_log_inst.get().info('Test: perplexity=%.3f, accuracy=%s' % (test_perplexity, accuracy))

        # acc compute
        '''
        for idx, domain_accu in enumerate(domain_accuracy):
            g_log_inst.get().info('Domain: %s, precision: %.3f, recall: %.3f' % (
            config.converter.label2domain[idx], domain_accuracy[idx][0] / float(domain_accuracy[idx][1]),
            domain_accuracy[idx][2] / float(domain_accuracy[idx][3])))
        '''
    g_log_inst.get().info('bilstm_attention training finished')


if __name__ == '__main__':
    g_log_inst.start('../log/train_emb.log', __name__, 'DEBUG')
    tf.app.run()
